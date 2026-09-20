"""
train_pipeline.py
=================
Pipeline completo de treinamento da Rede Neural Multitarefa Agronômica (PyTorch).
Executa:
- Particionamento estratificado Train/Validation/Test (80/10/10)
- Treinamento multitarefa com balanceamento de perdas
- Early stopping e salvamento de checkpoint do melhor modelo
- Avaliação detalhada de métricas:
  * Accuracy e Macro F1 para Irrigação
  * Subset Accuracy e F1-score para Pragas
  * MAE e RMSE (mm) para Lâmina de Água
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    classification_report
)

from data_engine import AgroDataEngine, AgroMultiTaskDataset
from model_multitask import AgroMultitaskNet


def train_multitask_model(
    csv_path: str = "dados_solo_2casas.csv",
    n_samples: int = 50000,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    model_save_path: str = "best_agro_multitask.pth",
    scaler_save_path: str = "agro_scaler_meta.pkl"
):
    """
    Executa o ciclo completo de treinamento, validação e teste.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f" Iniciando Pipeline de Treinamento AgroMultitaskNet", flush=True)
    print(f" Dispositivo de Computação: {device}", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Carregamento e Enriquecimento dos Dados
    engine = AgroDataEngine(scaler_path=scaler_save_path)
    df = engine.load_and_augment_dataset(csv_path=csv_path, n_samples=n_samples)

    # 2. Divisão Train / Validation / Test (80 / 10 / 10)
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["target_irrigacao"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["target_irrigacao"])

    print(f"Divisão dos dados concluída:", flush=True)
    print(f"  Treino (80%): {len(train_df):,} amostras", flush=True)
    print(f"  Validação (10%): {len(val_df):,} amostras", flush=True)
    print(f"  Teste (10%): {len(test_df):,} amostras\n", flush=True)

    # 3. Engenharia de Features e Normalização
    X_train = engine.prepare_feature_matrix(train_df, is_train=True)
    X_val = engine.prepare_feature_matrix(val_df, is_train=False)
    X_test = engine.prepare_feature_matrix(test_df, is_train=False)

    pragas_cols = ["target_fungos", "target_insetos", "target_acaros", "target_estresse"]

    ds_train = AgroMultiTaskDataset(
        X_train,
        train_df["target_irrigacao"].values,
        train_df[pragas_cols].values,
        train_df["target_volume_mm"].values
    )
    ds_val = AgroMultiTaskDataset(
        X_val,
        val_df["target_irrigacao"].values,
        val_df[pragas_cols].values,
        val_df["target_volume_mm"].values
    )
    ds_test = AgroMultiTaskDataset(
        X_test,
        test_df["target_irrigacao"].values,
        test_df[pragas_cols].values,
        test_df["target_volume_mm"].values
    )

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # 4. Instanciação do Modelo e Perdas
    in_features = X_train.shape[1]
    model = AgroMultitaskNet(in_features=in_features, num_irrig_classes=3, num_pest_labels=4).to(device)

    criterion_irrig = nn.CrossEntropyLoss()
    criterion_pragas = nn.BCEWithLogitsLoss()
    criterion_vol = nn.SmoothL1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    start_time = time.time()

    # 5. Loop de Treinamento
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_loss_irrig = 0.0
        train_loss_pragas = 0.0
        train_loss_vol = 0.0

        for bx, by_irrig, by_pragas, by_vol in loader_train:
            bx = bx.to(device)
            by_irrig = by_irrig.to(device)
            by_pragas = by_pragas.to(device)
            by_vol = by_vol.to(device)

            optimizer.zero_grad()

            pred_irrig, pred_pragas, pred_vol = model(bx)

            l_irrig = criterion_irrig(pred_irrig, by_irrig)
            l_pragas = criterion_pragas(pred_pragas, by_pragas)
            l_vol = criterion_vol(pred_vol, by_vol)

            # Combinação ponderada de perdas multitarefa
            loss = l_irrig + (1.2 * l_pragas) + (0.5 * l_vol)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_total += loss.item()
            train_loss_irrig += l_irrig.item()
            train_loss_pragas += l_pragas.item()
            train_loss_vol += l_vol.item()

        n_batches = len(loader_train)
        train_loss_total /= n_batches

        # Validação
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for bx, by_irrig, by_pragas, by_vol in loader_val:
                bx = bx.to(device)
                by_irrig = by_irrig.to(device)
                by_pragas = by_pragas.to(device)
                by_vol = by_vol.to(device)

                pred_irrig, pred_pragas, pred_vol = model(bx)
                l_irrig = criterion_irrig(pred_irrig, by_irrig)
                l_pragas = criterion_pragas(pred_pragas, by_pragas)
                l_vol = criterion_vol(pred_vol, by_vol)

                v_loss = l_irrig + (1.2 * l_pragas) + (0.5 * l_vol)
                val_loss_total += v_loss.item()

        val_loss_total /= len(loader_val)
        scheduler.step(val_loss_total)

        # Checkpoint do melhor modelo
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_features": in_features,
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }, model_save_path)
            salvo_msg = " [* MODELO SALVO]"
        else:
            salvo_msg = ""

        if epoch % 1 == 0 or epoch == epochs:
            print(f"Época [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss_total:.4f} | Val Loss: {val_loss_total:.4f} (Irrig: {train_loss_irrig/n_batches:.3f}, Pragas: {train_loss_pragas/n_batches:.3f}, Vol: {train_loss_vol/n_batches:.3f}){salvo_msg}", flush=True)

    tempo_total = time.time() - start_time
    print(f"\nTreinamento concluído em {tempo_total:.1f} segundos.", flush=True)

    # 6. Avaliação Final no Conjunto de Teste Cego (Test Set - 10%)
    print(f"\n=======================================================")
    print(f" Avaliação de Generalização no Conjunto de Teste Cego")
    print(f"=======================================================\n")

    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_pred_irrig = []
    all_true_irrig = []
    all_pred_pragas = []
    all_true_pragas = []
    all_pred_vol = []
    all_true_vol = []

    with torch.no_grad():
        for bx, by_irrig, by_pragas, by_vol in loader_test:
            bx = bx.to(device)
            out_irrig, out_pragas, out_vol = model(bx)

            all_pred_irrig.extend(torch.argmax(out_irrig, dim=-1).cpu().numpy())
            all_true_irrig.extend(by_irrig.numpy())

            all_pred_pragas.extend((torch.sigmoid(out_pragas) >= 0.5).cpu().numpy().astype(int))
            all_true_pragas.extend(by_pragas.numpy())

            all_pred_vol.extend(out_vol.cpu().numpy().flatten())
            all_true_vol.extend(by_vol.numpy().flatten())

    all_pred_irrig = np.array(all_pred_irrig)
    all_true_irrig = np.array(all_true_irrig)
    all_pred_pragas = np.array(all_pred_pragas)
    all_true_pragas = np.array(all_true_pragas)
    all_pred_vol = np.array(all_pred_vol)
    all_true_vol = np.array(all_true_vol)

    # Métricas Tarefa 1: Irrigação
    acc_irrig = accuracy_score(all_true_irrig, all_pred_irrig)
    f1_irrig = f1_score(all_true_irrig, all_pred_irrig, average="macro")

    print(f"[TAREFA 1 - DECISÃO DE IRRIGAÇÃO]")
    print(f"  Acurácia Geral: {acc_irrig * 100:.2f}%")
    print(f"  Macro F1-Score: {f1_irrig:.4f}")
    target_names = ["0: Não Irrigar", "1: Leve/Moderada", "2: Abundante/Turno"]
    print(classification_report(all_true_irrig, all_pred_irrig, target_names=target_names, digits=3))

    # Métricas Tarefa 2: Pragas e Doenças
    f1_pragas_micro = f1_score(all_true_pragas, all_pred_pragas, average="micro", zero_division=0)
    f1_pragas_macro = f1_score(all_true_pragas, all_pred_pragas, average="macro", zero_division=0)
    subset_acc = accuracy_score(all_true_pragas, all_pred_pragas)

    print(f"[TAREFA 2 - RISCO DE PRAGAS E FITOSSANIDADE]")
    print(f"  Subset Accuracy (Correspondência Exata 4 labels): {subset_acc * 100:.2f}%")
    print(f"  Micro F1-Score: {f1_pragas_micro:.4f}")
    print(f"  Macro F1-Score: {f1_pragas_macro:.4f}")
    for i, col in enumerate(pragas_cols):
        acc_label = accuracy_score(all_true_pragas[:, i], all_pred_pragas[:, i])
        f1_label = f1_score(all_true_pragas[:, i], all_pred_pragas[:, i], zero_division=0)
        print(f"   - {col.replace('target_', '').capitalize()}: Acurácia = {acc_label * 100:.2f}%, F1 = {f1_label:.4f}")

    # Métricas Tarefa 3: Volume de Água (mm)
    mae_vol = mean_absolute_error(all_true_vol, all_pred_vol)
    rmse_vol = root_mean_squared_error(all_true_vol, all_pred_vol)

    print(f"\n[TAREFA 3 - LÂMINA DE ÁGUA NECESSÁRIA (mm)]")
    print(f"  MAE (Erro Médio Absoluto): {mae_vol:.3f} mm")
    print(f"  RMSE (Raiz do Erro Quadrático Médio): {rmse_vol:.3f} mm")
    print(f"=======================================================\n")

    return {
        "acc_irrig": acc_irrig,
        "f1_irrig": f1_irrig,
        "f1_pragas_macro": f1_pragas_macro,
        "subset_acc_pragas": subset_acc,
        "mae_volume_mm": mae_vol,
        "rmse_volume_mm": rmse_vol
    }


if __name__ == "__main__":
    train_multitask_model()
