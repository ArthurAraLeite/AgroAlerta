"""
agronomy_rules.py
=================
Módulo de regras agronômicas de precisão, constantes físicas do solo,
balanço hídrico (FAO-56), cálculo de déficit, fitossanidade e catálogo
de insumos com separação estrita entre Cultivo Orgânico e Convencional.
"""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class CulturaProfile:
    nome: str
    kc: float
    ponto_murcha_pct: float
    profundidade_raiz_m: float
    sensibilidade_estresse: float  # fator p (fração de esgotamento)


@dataclass
class SoloProfile:
    tipo: str
    capacidade_campo_pct: float
    ponto_critico_pct: float
    densidade_aparente_g_cm3: float


# Perfis agronômicos de culturas
CULTURAS_CONFIG: Dict[str, CulturaProfile] = {
    "milho": CulturaProfile(
        nome="milho",
        kc=1.15,
        ponto_murcha_pct=2.8,
        profundidade_raiz_m=0.60,
        sensibilidade_estresse=0.55
    ),
    "soja": CulturaProfile(
        nome="soja",
        kc=1.10,
        ponto_murcha_pct=2.5,
        profundidade_raiz_m=0.50,
        sensibilidade_estresse=0.50
    ),
    "tomate": CulturaProfile(
        nome="tomate",
        kc=1.15,
        ponto_murcha_pct=3.2,
        profundidade_raiz_m=0.45,
        sensibilidade_estresse=0.40
    ),
    "alface": CulturaProfile(
        nome="alface",
        kc=0.95,
        ponto_murcha_pct=3.8,
        profundidade_raiz_m=0.25,
        sensibilidade_estresse=0.30
    ),
    "cafe": CulturaProfile(
        nome="cafe",
        kc=0.90,
        ponto_murcha_pct=2.2,
        profundidade_raiz_m=0.80,
        sensibilidade_estresse=0.60
    )
}

# Perfis físicos dos solos (compatíveis com o domínio de umidade 1.0% a 10.0% do dataset)
SOLO_CONFIG: Dict[str, SoloProfile] = {
    "arenoso": SoloProfile(
        tipo="arenoso",
        capacidade_campo_pct=4.8,
        ponto_critico_pct=2.2,
        densidade_aparente_g_cm3=1.55
    ),
    "ideal": SoloProfile(
        tipo="ideal",
        capacidade_campo_pct=7.2,
        ponto_critico_pct=3.6,
        densidade_aparente_g_cm3=1.30
    ),
    "argiloso": SoloProfile(
        tipo="argiloso",
        capacidade_campo_pct=9.5,
        ponto_critico_pct=4.9,
        densidade_aparente_g_cm3=1.18
    )
}

# Catálogo oficial de insumos e manejos autorizados
MANEJO_CATALOGO = {
    "organico": {
        "fungos_bacterias": [
            "Aplicação foliar preventiva de Calda Bordalesa (concentração 1% p/v)",
            "Pulverização de Calda Sulfocálcica (diluição 1:60 a 1:100)",
            "Aplicação biológica de Trichoderma harzianum / Trichoderma asperellum via solo e folíolo",
            "Manejo cultural: desbrota, raleio e aumento do espaçamento para ventilação do dossel"
        ],
        "insetos_lagartas": [
            "Aplicação de inseticida biológico à base de Bacillus thuringiensis (Bt) - estirpes kurstaki",
            "Pulverização de Óleo de Neem 100% puro prensado a frio (0.5% a 1.0% de calda)",
            "Instalação de armadilhas luminosas e feromônios específicos de captura",
            "Liberação inundativa de parasitoides de ovos (Trichogramma pretiosum)"
        ],
        "acaros_tripes": [
            "Aplicação de bioacaricida à base do fungo entomopatogênico Beauveria bassiana",
            "Pulverização em bordadura com sabão potássico vegetal neutro (1.5%) + Extrato Pirolenhoso (0.3%)",
            "Manejo de umidade e eliminação de plantas hospedeiras espontâneas adjacentes"
        ],
        "estresse_hidrico_salino": [
            "Cobertura morta (mulching) de palhada densa para retenção hídrica e amortecimento térmico",
            "Incorporação de Húmus de Minhoca / Composto Orgânico estabilizado para elevar CTC do solo",
            "Irrigação por gotejamento pulsado para lixiviação suave de sais"
        ],
        "nutricao_npk": [
            "Adubação fosfatada via Fosfato Natural Reativo (ex: Bayóvar) / Farinha de Ossos calcinada",
            "Suprimento potássico com Cinzas vegetais de biomassa / Pó de Rocha Silicatado (Remineralizador)",
            "Suprimento de Nitrogênio com Torta de Mamona desintoxicada e biofertilizante aeróbico enriquecido",
            "Aplicação de calcário dolomítico e gesso agrícola para correção de saturação por bases"
        ]
    },
    "convencional": {
        "fungos_bacterias": [
            "Aplicação de fungicidas sistêmicos do grupo dos Triazóis + Estrobilurinas (ex: Azoxistrobina + Ciproconazol)",
            "Pulverização protetora multissítio com Mancozeb ou Oxicloreto de Cobre",
            "Aplicação curativa de Clorotalonil em caso de lesões ativas"
        ],
        "insetos_lagartas": [
            "Aplicação de inseticida seletivo do grupo das Diamidas Antranílicas (Clorantraniliprole)",
            "Pulverização de choque com Piretroides de amplo espectro (Lambda-cialotrina / Bifentrina)",
            "Uso de Neonicotinoides via drench ou tratamento de sementes"
        ],
        "acaros_tripes": [
            "Aplicação de acaricida específico de ação translaminar (Abamectina ou Espirodiclofeno)",
            "Rotacionamento químico com Fenpiroximato para quebra de resistência de populações"
        ],
        "estresse_hidrico_salino": [
            "Lâmina de lixiviação calculada para fração de eluição de sais (Leaching Fraction de 15%)",
            "Condicionamento físico do solo com subsolagem / descompactação mecânica entrelinhas",
            "Fertirrigação com biopolímeros anti-stress hídrico e aminoácidos livres"
        ],
        "nutricao_npk": [
            "Fertirrigação solúvel: Nitrato de Cálcio, Fosfato Monoamônico (MAP solúvel) e Sulfato de Potássio",
            "Adubação mineral nitrogenada de cobertura: Ureia com inibidor de urease (NBPT) ou Nitrato de Amônio",
            "Correção de micronutrientes com quelatos sintéticos (EDTA / EDDHA)"
        ]
    }
}


def calcular_balanco_hidrico_agronomico(
    tipo_solo: str,
    umidade_atual_pct: float,
    cultura_nome: str,
    compactacao_kpa: float,
    et0_fao_12h: float,
    rain_6h: float,
    rain_12h: float,
    precip_prob_max_6h: float
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Calcula a necessidade de irrigação e lâmina de água (mm) com base nas
    leis de balanço hídrico FAO-56 e regras de negócio com ponderação agronômica.

    Ponderação:
    - 40% Déficit de Umidade Atual do Solo
    - 35% Balanço Hídrico Futuro curto prazo (Chuva esperada nas próximas 12h - ETc)
    - 25% Fator da Cultura (Kc) e Compactação do Solo
    
    Trava de Segurança Estrita:
    - Se precip_prob_max_6h > 70% ou rain_6h > 5.0mm -> SUSPENDER (classe 0, volume 0.0mm).
    """
    cultura = CULTURAS_CONFIG.get(cultura_nome.lower(), CULTURAS_CONFIG["milho"])
    solo = SOLO_CONFIG.get(tipo_solo.lower(), SOLO_CONFIG["ideal"])

    # 1. Trava estrita de segurança meteorológica
    trava_chuva = (precip_prob_max_6h > 70.0) or (rain_6h >= 5.0)
    if trava_chuva:
        return 0, 0.0, {
            "trava_chuva_acionada": True,
            "motivo_trava": f"Precipitação iminente detectada (Prob 6h: {precip_prob_max_6h:.1f}%, Chuva 6h: {rain_6h:.1f}mm). Irrigação suspensa.",
            "deficit_solo_pct": max(0.0, solo.capacidade_campo_pct - umidade_atual_pct),
            "etc_12h_mm": et0_fao_12h * cultura.kc,
            "balanco_futuro_mm": rain_12h - (et0_fao_12h * cultura.kc),
            "fator_ponderado": 0.0
        }

    # 2. Componentes de decisão
    deficit_umidade_pct = max(0.0, solo.capacidade_campo_pct - umidade_atual_pct)
    faixa_disponivel = max(0.1, solo.capacidade_campo_pct - solo.ponto_critico_pct)
    score_deficit = min(1.0, deficit_umidade_pct / (faixa_disponivel * 1.5))

    etc_12h = max(0.0, et0_fao_12h * cultura.kc)
    deficit_atmosferico_mm = max(0.0, etc_12h - rain_12h)
    score_balanco = min(1.0, deficit_atmosferico_mm / 6.0)

    score_kc = (cultura.kc - 0.4) / (1.25 - 0.4)
    score_compactacao = min(1.0, max(0.0, (compactacao_kpa - 1000.0) / 2000.0))
    score_contexto = 0.6 * score_kc + 0.4 * (1.0 - 0.3 * score_compactacao)

    # 3. Combinação Ponderada (40% Déficit Solo, 35% Balanço Futuro, 25% Fator Cultura/Compactação)
    indice_rega = (0.40 * score_deficit) + (0.35 * score_balanco) + (0.25 * score_contexto)

    # 4. Cálculo da Lâmina Líquida Real (mm)
    profundidade_efetiva_dm = cultura.profundidade_raiz_m * 10.0
    lamina_fisica_solo_mm = max(0.0, (solo.capacidade_campo_pct - umidade_atual_pct) * solo.densidade_aparente_g_cm3 * profundidade_efetiva_dm * 0.1)
    lamina_liquida_mm = max(0.0, lamina_fisica_solo_mm + deficit_atmosferico_mm)

    if umidade_atual_pct >= solo.capacidade_campo_pct and rain_12h >= etc_12h:
        indice_rega = 0.0
        lamina_liquida_mm = 0.0

    # 5. Categorização da decisão
    if indice_rega < 0.30 or lamina_liquida_mm < 1.0:
        classe_irrigacao = 0  # Não Irrigar
        lamina_liquida_mm = 0.0
    elif indice_rega < 0.65 or lamina_liquida_mm < 7.0:
        classe_irrigacao = 1  # Irrigar Leve/Moderada
        lamina_liquida_mm = min(6.5, max(1.5, lamina_liquida_mm))
    else:
        classe_irrigacao = 2  # Irrigar Abundante / Turno Completo
        lamina_liquida_mm = min(25.0, max(7.0, lamina_liquida_mm))

    return classe_irrigacao, round(lamina_liquida_mm, 2), {
        "trava_chuva_acionada": False,
        "indice_rega_ponderado": round(indice_rega, 4),
        "deficit_umidade_pct": round(deficit_umidade_pct, 2),
        "etc_12h_mm": round(etc_12h, 2),
        "deficit_atmosferico_mm": round(deficit_atmosferico_mm, 2),
        "lamina_calculada_mm": round(lamina_liquida_mm, 2)
    }


def calcular_riscos_patogenos_clima(
    temperatura_2m: float,
    umidade_relativa_2m: float,
    umidade_solo_pct: float,
    condutividade_eletrica: float,
    tipo_solo: str,
    cultura_nome: str
) -> Tuple[List[int], Dict[str, float]]:
    """
    Avalia os riscos fitossanitários com base no microclima e solo:
    0: Risco de Fungos/Bactérias
    1: Risco de Insetos/Lagartas
    2: Risco de Ácaros/Tripes
    3: Risco de Estresse Hídrico/Salino
    """
    cultura = CULTURAS_CONFIG.get(cultura_nome.lower(), CULTURAS_CONFIG["milho"])
    solo = SOLO_CONFIG.get(tipo_solo.lower(), SOLO_CONFIG["ideal"])

    # 1. Fungos: UR > 80% e 20°C <= Temp <= 28°C
    cond_fungo_temp = 18.0 <= temperatura_2m <= 28.5
    cond_fungo_ur = umidade_relativa_2m >= 78.0
    prob_fungos = 0.0
    if cond_fungo_temp and cond_fungo_ur:
        prob_fungos = 0.85 + 0.15 * min(1.0, (umidade_relativa_2m - 78.0) / 22.0)
    elif cond_fungo_ur or (20.0 <= temperatura_2m <= 27.0 and umidade_relativa_2m >= 65.0):
        prob_fungos = 0.45
    else:
        prob_fungos = 0.10

    # 2. Ácaros e Tripes: Clima Quente e Seco (Temp > 30°C e UR < 40%)
    prob_acaros = 0.0
    if temperatura_2m >= 29.5 and umidade_relativa_2m <= 45.0:
        prob_acaros = 0.85 + 0.15 * min(1.0, (temperatura_2m - 29.5) / 10.0)
    elif temperatura_2m >= 28.0 and umidade_relativa_2m <= 55.0:
        prob_acaros = 0.45
    else:
        prob_acaros = 0.08

    # 3. Insetos Mastigadores / Lagartas: 22°C a 32°C com umidade moderada (45% a 75%)
    prob_insetos = 0.0
    if 22.0 <= temperatura_2m <= 32.0 and 45.0 <= umidade_relativa_2m <= 75.0:
        prob_insetos = 0.70
    elif 20.0 <= temperatura_2m <= 35.0:
        prob_insetos = 0.35
    else:
        prob_insetos = 0.12

    # 4. Estresse Hídrico ou Salino:
    prob_estresse = 0.0
    estresse_salino = condutividade_eletrica >= 3.0
    estresse_hidrico = umidade_solo_pct <= cultura.ponto_murcha_pct
    if estresse_salino or estresse_hidrico:
        prob_estresse = 0.90
    elif condutividade_eletrica >= 2.0 or umidade_solo_pct <= solo.ponto_critico_pct:
        prob_estresse = 0.50
    else:
        prob_estresse = 0.05

    labels = [
        1 if prob_fungos >= 0.5 else 0,
        1 if prob_insetos >= 0.5 else 0,
        1 if prob_acaros >= 0.5 else 0,
        1 if prob_estresse >= 0.5 else 0,
    ]

    probs = {
        "prob_fungos": round(prob_fungos, 3),
        "prob_insetos": round(prob_insetos, 3),
        "prob_acaros": round(prob_acaros, 3),
        "prob_estresse": round(prob_estresse, 3)
    }

    return labels, probs


def diagnosticar_solo(
    ph: float,
    nitrogenio: float,
    fosforo: float,
    potassio: float,
    condutividade: float,
    compactacao: float
) -> Dict[str, Any]:
    """
    Gera diagnóstico agronômico e faixas de conformidade físico-química do solo.
    """
    if ph < 5.5:
        ph_status = "Ácido (Necessidade de Calagem para neutralizar Alumínio e elevar saturação por bases)"
    elif ph <= 6.8:
        ph_status = "Ideal / Equilibrado (Faixa de máxima disponibilidade nutricional)"
    else:
        ph_status = "Alcalino (Risco de indisponibilidade de micronutrientes catiônicos Fe, Zn, Mn)"

    if condutividade < 1.0:
        ce_status = "Baixa salinidade (Normal para a maioria das culturas)"
    elif condutividade <= 2.5:
        ce_status = "Salinidade moderada (Atenção para culturas sensíveis como alface e tomate)"
    else:
        ce_status = "Solo Salinizado (Elevada pressão osmótica prejudicial à absorção de água)"

    if compactacao < 1500:
        comp_status = "Solo Descompactado (Excelente aeração e crescimento radicular profundo)"
    elif compactacao <= 2200:
        comp_status = "Compactação Moderada (Monitorar tráfego de maquinário)"
    else:
        comp_status = "Solo Altamente Compactado (>2 MPa) - Impedimento mecânico severo"

    return {
        "ph": {"valor": ph, "status": ph_status},
        "nitrogenio_n_ppm": {"valor": nitrogenio, "status": "Baixo" if nitrogenio < 3.0 else ("Adequado" if nitrogenio <= 7.0 else "Alto")},
        "fosforo_p_ppm": {"valor": fosforo, "status": "Baixo" if fosforo < 2.5 else ("Adequado" if fosforo <= 6.0 else "Alto")},
        "potassio_k_ppm": {"valor": potassio, "status": "Baixo" if potassio < 2.0 else ("Adequado" if potassio <= 5.0 else "Alto")},
        "condutividade_eletrica_dS_m": {"valor": condutividade, "status": ce_status},
        "compactacao_kpa": {"valor": compactacao, "status": comp_status}
    }
