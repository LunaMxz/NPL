""" Configuración de la Etapa 1. Parametros ajustables. """

#Read
COLUMNAS = ["id", "materia", "sentimiento", "comentario"]
DELIMITADOR = "auto"
ENCODING_LECTURA = "utf-8"

#clean
CLASES_VALIDAS = ["positive", "negative", "neutral", "irrelevant"]
UMBRAL_CASI_DUPLICADO = 85
MIN_TOKENS_UTILES = 2

CATALOGO_MATERIAS = [
    "Agile Project Management", "Big Data Analytics", "Cloud Computing Essentials",
    "Computer Networks 101", "Cybersecurity Introduction", "Data Visualization with Power BI",
    "Database Systems Advanced", "Deep Learning Fundamentals", "Digital Marketing Strategy",
    "Intro to AI Ethics", "Java Programming Zero-to-Hero", "Machine Learning with TensorFlow",
    "Mobile App Development with Flutter", "Natural Language Processing",
    "Python for Data Science", "React Frontend Mastery", "Software Engineering Principles",
    "UI/UX Design Basics", "Web Development Bootcamp",
]

MATERIAS_NO_ACADEMICAS = [
    "Amazon", "ApexLegends", "AssassinsCreed", "Battlefield", "Borderlands",
    "CallOfDuty", "CallOfDutyBlackopsColdWar", "CS-GO", "Cyberpunk2077",
    "Dota2", "Facebook", "FIFA", "Fortnite", "Google", "GrandTheftAuto(GTA)",
    "Hearthstone", "HomeDepot", "johnson&johnson", "LeagueOfLegends",
    "MaddenNFL", "Microsoft", "NBA2K", "Nvidia", "Overwatch",
    "PlayerUnknownsBattlegrounds(PUBG)", "PlayStation5(PS5)",
    "RedDeadRedemption(RDR)", "TomClancysGhostRecon", "TomClancysRainbowSix",
    "Verizon", "WorldOfCraft",
]
#estadisticas (nombres de los archivos de salida)
ARCHIVO_LIMPIO = "feedback_training_limpio.csv"
ARCHIVO_ESTADISTICAS = "estadisticas_etapa1.json"
ARCHIVO_POR_MATERIA = "estadisticas_por_materia.csv"
ARCHIVO_TEXTOS_REPETIDOS = "textos_repetidos.csv"
TOP_TEXTOS_REPETIDOS = 50
