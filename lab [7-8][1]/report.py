# Скрипт для анализа онтологии
from rdflib import Graph
import pandas as pd

def analyze_ontology(file_path):
    g = Graph()
    g.parse(file_path, format="turtle")

    # Статистика онтологии
    classes = list(g.subjects(predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", object="http://www.w3.org/2002/07/owl#Class"))

    properties = list(g.subjects(predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", object="http://www.w3.org/2002/07/owl#ObjectProperty"))
    
    individuals = list(g.subjects(predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", object="http://www.w3.org/2002/07owl#NamedIndividual"))

    print(f"Классы: {len(classes)}")
    print(f"Свойства: {len(properties)}")
    print(f"Индивиды: {len(individuals)}")
    
    
    return {
        "classes": len(classes),
        "properties": len(properties),
        "individuals": len(individuals)
    }

# Анализ оригинальной и модифицированной онтологии
stats_original = analyze_ontology("pizza.ttl")
stats_modified = analyze_ontology("pizza_russian.ttl")

# Создание отчета
report = pd.DataFrame([stats_original, stats_modified], index=["Original", "Modified"])
report.to_csv("ontology_report.csv")