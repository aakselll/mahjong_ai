"""
Predicteur de reward Phi (section 3.2 du papier Suphx) : un GRU a 2 couches
suivi de 2 couches fully-connected, qui apprend a predire le reward final de
partie (base sur le classement) a partir de l'historique des manches jouees.

Utilise ensuite comme Phi(x^k) - Phi(x^{k-1}) pour attribuer un reward a
chaque manche pendant l'entrainement RL (cf. training/game_env.py), au lieu
du score brut de la manche -- ce qui permet a l'agent d'apprendre des
comportements qui dependent du contexte de la partie (ex: jouer defensif
avec une grosse avance) sans que ce ne soit code en dur dans les regles.

[Amelioration] Ajout d'un dropout leger (GRU inter-couches + tete FC) et
d'une tranche de validation dans train_reward_predictor(). Avec un buffer
relativement petit et des sequences correlees (les 4 joueurs d'une meme
partie ne sont pas des echantillons independants), c'est le composant du
pipeline le plus expose a un vrai surapprentissage (au sens classique) --
et jusqu'ici rien ne permettait de le mesurer : seule la loss d'ENTRAINEMENT
etait loggee. Mettre dropout=0.0 et val_fraction=0.0 restaure le
comportement d'origine.
"""
import numpy as np
import torch
import torch.nn as nn

FEATURE_DIM = 8  # taille du vecteur de features construit dans game_env.py (par manche, par joueur)


class RewardPredictor(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # nn.GRU n'applique dropout qu'ENTRE les couches -> sans effet si num_layers=1
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: (B, T, input_dim) -> predictions (B, T), UNE par pas de temps
        (necessaire pour la loss de l'eq. 4 du papier : comparer Phi(x^1..x^j)
        au reward final R pour TOUS les prefixes j, pas seulement le dernier
        -- c'est ce qui donne un signal d'entrainement a chaque manche)."""
        h, _ = self.gru(x)
        return self.fc(h).squeeze(-1)

    @torch.no_grad()
    def predict_step(self, feat, hidden_state):
        """Inference incrementale pour UNE manche, en portant l'etat cache du
        GRU d'un appel a l'autre (evite de rejouer toute la partie a chaque
        manche). feat: np.array (FEATURE_DIM,). hidden_state: tensor
        (num_layers, 1, hidden_dim), ou None pour la toute premiere manche
        d'une partie (nn.GRU initialise alors a zero automatiquement).

        Force le mode eval() : avec le dropout ajoute ci-dessus, l'inference
        utilisee pour attribuer un reward RL doit rester deterministe -- le
        dropout ne doit jouer son role de regularisation que pendant
        train_reward_predictor(), jamais ici."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(feat, dtype=torch.float32, device=device).view(1, 1, -1)
        h_out, new_hidden = self.gru(x, hidden_state)
        pred = self.fc(h_out[:, -1, :]).squeeze().item()
        return pred, new_hidden


def train_reward_predictor(model, optimizer, buffer, batch_size=256, epochs=2, device="cpu",
                            val_fraction=0.1):
    """Entrainement supervise (eq. 4 du papier). Toutes les sequences du
    buffer ont la meme longueur fixe (N_ROUNDS, cf. game_env.py -- pas de
    rotation dealer-sur-victoire dans notre simplification), donc pas besoin
    de padding/masking pour les batcher.

    val_fraction : fraction du buffer mise de cote pour la validation, en
    prenant les entrees les plus RECEMMENT ajoutees (le buffer est une file
    glissante FIFO, cf. GameLogBuffer.add) plutot qu'un split aleatoire --
    ca reste une validation approximative (les frontieres train/val se
    deplacent d'un appel a l'autre a mesure que le buffer avance), mais ca
    donne un signal bien plus utile que zero validation : si val_loss se
    met a monter alors que train_loss continue de baisser, c'est le signe
    que Phi surapprend le buffer plutot que de generaliser.

    Renvoie (train_loss, val_loss) -- l'un ou l'autre peut etre None si le
    buffer est trop petit pour le constituer."""
    if len(buffer) < max(batch_size, 2):
        return None, None

    sequences = np.stack(buffer.sequences, axis=0)          # (N, T, FEATURE_DIM)
    targets = np.array(buffer.final_rewards, dtype=np.float32)  # (N,)

    N = len(sequences)
    n_val = max(int(N * val_fraction), 1) if val_fraction > 0 else 0
    val_idx = np.arange(N - n_val, N) if n_val > 0 else np.array([], dtype=int)
    train_idx = np.arange(0, N - n_val)

    total_loss, n_updates = 0.0, 0
    model.train()
    for _ in range(epochs):
        np.random.shuffle(train_idx)
        for start in range(0, len(train_idx), batch_size):
            mb = train_idx[start:start + batch_size]
            if len(mb) < 2:
                continue
            x = torch.as_tensor(sequences[mb], dtype=torch.float32, device=device)
            r = torch.as_tensor(targets[mb], dtype=torch.float32, device=device)

            preds = model(x)                          # (B, T)
            r_expanded = r.unsqueeze(1).expand_as(preds)  # meme cible R a chaque pas de temps
            loss = ((preds - r_expanded) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_updates += 1

    train_loss = total_loss / max(n_updates, 1)

    val_loss = None
    if len(val_idx) >= 2:
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(sequences[val_idx], dtype=torch.float32, device=device)
            r = torch.as_tensor(targets[val_idx], dtype=torch.float32, device=device)
            preds = model(x)
            r_expanded = r.unsqueeze(1).expand_as(preds)
            val_loss = ((preds - r_expanded) ** 2).mean().item()
        model.train()

    return train_loss, val_loss
