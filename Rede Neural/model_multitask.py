"""
model_multitask.py
==================
Arquitetura da Rede Neural Multitarefa (Multitask Deep Learning) em PyTorch
para tomada de decisões agrícolas de alta precisão.
Possui um Shared Backbone e 3 cabeças de predição:
1. Irrigação (Classificação Multiclasse: 3 classes)
2. Risco de Pragas/Doenças (Classificação Multilabel: 4 labels)
3. Volume de Água em mm (Regressão Contínua: Lâmina líquida)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AgroMultitaskNet(nn.Module):
    """
    Rede Neural Multitarefa para Agricultura de Precisão.
    Combina atributos do solo (físico-químicos), dados de IoT e variáveis meteorológicas
    para inferir simultaneamente necessidade de manejo hídrico e fitossanitário.
    """

    def __init__(self, in_features: int, num_irrig_classes: int = 3, num_pest_labels: int = 4):
        super(AgroMultitaskNet, self).__init__()

        # Tronco Compartilhado (Shared Backbone)
        self.shared_block1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30)
        )

        self.shared_block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20)
        )

        self.shared_block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Cabeça 1: Decisão de Irrigação (Multiclasse)
        self.head_irrigacao = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_irrig_classes)  # Retorna Logits para CrossEntropyLoss
        )

        # Cabeça 2: Risco de Pragas e Doenças (Multilabel)
        self.head_pragas = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_pest_labels)   # Retorna Logits para BCEWithLogitsLoss
        )

        # Cabeça 3: Volume de Água Necessário em mm (Regressão)
        self.head_volume = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.ReLU()  # Lâmina hídrica não-negativa (mm >= 0)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executa a passagem para frente retornando os logits de cada tarefa:
        - logits_irrig: (Batch, 3)
        - logits_pragas: (Batch, 4)
        - volume_mm: (Batch, 1)
        """
        # Shared representations
        h1 = self.shared_block1(x)
        h2 = self.shared_block2(h1)
        features = self.shared_block3(h2)

        # Task-specific outputs
        logits_irrig = self.head_irrigacao(features)
        logits_pragas = self.head_pragas(features)
        volume_mm = self.head_volume(features)

        return logits_irrig, logits_pragas, volume_mm

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Aplica ativações finais (Softmax e Sigmoid) para inferência em produção.
        """
        self.eval()
        logits_irrig, logits_pragas, volume_mm = self.forward(x)

        probs_irrig = F.softmax(logits_irrig, dim=-1)
        probs_pragas = torch.sigmoid(logits_pragas)

        return {
            "probs_irrigacao": probs_irrig,
            "probs_pragas": probs_pragas,
            "volume_mm": volume_mm.squeeze(-1)
        }
