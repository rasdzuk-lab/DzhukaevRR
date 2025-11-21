from rdflib import Graph, URIRef, RDF, OWL, RDFS
import pandas as pd

def analyze_ontology(file_path):
    g = Graph()
    
    # Определяем формат по расширению файла
    if file_path.endswith('.ttl'):
        g.parse(file_path, format="turtle")
    elif file_path.endswith('.rdf') or file_path.endswith('.owl'):
        g.parse(file_path, format="xml")
    else:
        g.parse(file_path)  # Автоопределение
    
    # Более надежный подсчет классов
    owl_class = URIRef("http://www.w3.org/2002/07/owl#Class")
    
    # Находим все объявленные классы
    declared_classes = set()
    for s in g.subjects(RDF.type, owl_class):
        declared_classes.add(s)
    
    # Также находим классы через rdfs:subClassOf (иерархия)
    for s in g.subjects(RDFS.subClassOf, None):
        declared_classes.add(s)
    
    # Объектные свойства
    object_property = URIRef("http://www.w3.org/2002/07/owl#ObjectProperty")
    properties = list(g.subjects(RDF.type, object_property))
    
    # Индивиды
    named_individual = URIRef("http://www.w3.org/2002/07/owl#NamedIndividual")
    individuals = list(g.subjects(RDF.type, named_individual))
    
    print(f"Классы: {len(declared_classes)}")
    print(f"Свойства: {len(properties)}")
    print(f"Индивиды: {len(individuals)}")
    
    # Дополнительная информация
    print(f"Всего триплов в онтологии: {len(g)}")
    
    return {
        "classes": len(declared_classes),
        "properties": len(properties),
        "individuals": len(individuals),
        "triples": len(g)
    }

# Тестируем на файле
try:
    stats_modified = analyze_ontology("pizza_russian.rdf")
    print("Анализ успешно завершен!")
except Exception as e:
    print(f"Ошибка при анализе: {e}")

# Для сравнения с оригиналом (если есть)
try:
    stats_original = analyze_ontology("pizza.owl")
    
    # Создание отчета
    report = pd.DataFrame([stats_original, stats_modified], 
                         index=["Original", "Modified"])
    report.to_csv("ontology_report.csv")
    print("Отчет создан: ontology_report.csv")
except FileNotFoundError:
    print("Оригинальный файл pizza.owl не найден, создаю отчет только для модифицированной онтологии")
    report = pd.DataFrame([stats_modified], index=["Modified"])
    report.to_csv("ontology_report.csv")