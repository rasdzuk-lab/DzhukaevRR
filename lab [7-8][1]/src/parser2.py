from rdflib import Graph
import pandas as pd

def simple_ontology_analysis(file_path):
    g = Graph()
    
    # Парсим с автоопределением формата
    g.parse(file_path)
    
    # Простой подсчет через SPARQL запрос
    query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    SELECT (COUNT(DISTINCT ?class) as ?classes) 
           (COUNT(DISTINCT ?prop) as ?properties) 
           (COUNT(DISTINCT ?ind) as ?individuals)
    WHERE {
      { ?class a owl:Class }
      UNION
      { ?prop a owl:ObjectProperty }
      UNION  
      { ?ind a owl:NamedIndividual }
    }
    """
    
    result = g.query(query)
    for row in result:
        classes = row.classes
        properties = row.properties
        individuals = row.individuals
    
    print(f"Классы: {classes}")
    print(f"Свойства: {properties}")
    print(f"Индивиды: {individuals}")
    
    return {
        "classes": classes,
        "properties": properties,
        "individuals": individuals
    }

# Использование
stats = simple_ontology_analysis("pizza_russian.rdf")