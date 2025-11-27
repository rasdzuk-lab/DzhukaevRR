from SPARQLWrapper import SPARQLWrapper, JSON, XML
import pandas as pd

# Настройка SPARQL endpoint
sparql = SPARQLWrapper("http://localhost:3030/pizza-ds/sparql")
sparql.setReturnFormat(JSON)

def run_query(query):
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return results
    except Exception as e:
        print(f"Ошибка выполнения запроса: {e}")
        return None

query1 = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?class ?label
WHERE {
    ?class a owl:Class .
    OPTIONAL { ?class rdfs:label ?label }
}
ORDER BY ?class
"""

results1 = run_query(query1)
print("Классы онтологии:")
for result in results1["results"]["bindings"]:
    print(f"{result['class']['value']} - {result.get('label', {}).get('value', 'No label')}")

query2 = """
PREFIX pizza: <http://www.co-ode.org/ontologies/pizza/pizza.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?pizza ?name
WHERE {
  	{
    	?pizza owl:equivalentClass ?Class .
    	?Class owl:intersectionOf ?intersec .
    	?intersec ?about pizza:Pizza .
  		?pizza rdfs:label ?name .
 	}
  	UNION
  	{
  		?pizza rdfs:subClassOf pizza:NamedPizza .
    	?pizza rdfs:label ?name .
  	}
}
ORDER BY ?name
"""

results2 = run_query(query2)
print("\nВсе пиццы:")
for result in results2["results"]["bindings"]:
    print(result['name']['value'])

query3 = """
PREFIX pizza: <http://www.co-ode.org/ontologies/pizza/pizza.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?pizza ?name ?topping
WHERE {
  	{
    	?pizza owl:equivalentClass ?Class .
    	?Class owl:intersectionOf ?intersec .
    	?intersec ?about pizza:Pizza .
  		?pizza rdfs:label ?name .
 	}
  	UNION
  	{
  		?pizza rdfs:subClassOf pizza:NamedPizza .
    	?pizza rdfs:label ?name .
  	}
  	?pizza rdfs:subClassOf ?Restriction .
  	?Restriction owl:someValuesFrom ?topping .
  	FILTER STRENDS(STR(?topping), "MushroomTopping")
    FILTER (lang(?name) = "en")
}
"""

results3 = run_query(query3)
print("\nПиццы с грибами:")
for result in results3["results"]["bindings"]:
    print(f"{result['name']['value']} - {result['topping']['value']}")

query4 = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX pizza: <http://www.co-ode.org/ontologies/pizza/pizza.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?topping (COUNT(?pizza) AS ?count)
WHERE {
  	{
    	?pizza owl:equivalentClass ?Class .
    	?Class owl:intersectionOf ?intersec .
    	?intersec ?about pizza:Pizza .
 	}
  	UNION
  	{
  		?pizza rdfs:subClassOf pizza:NamedPizza .
  	}
    # Получение начинок
    {
  		?pizza owl:equivalentClass ?Class .
    	?Class owl:intersectionOf ?intersec .
  		?intersec rdf:rest*/rdf:first ?Restriction .
  		?Restriction owl:someValuesFrom ?topping .
  	}
  	UNION
  	{
  		?pizza rdfs:subClassOf ?Restriction .
  		?Restriction owl:someValuesFrom ?topping .
  	}
}
GROUP BY ?topping
ORDER BY DESC(?count)
LIMIT 10
"""
results4 = run_query(query4)
print("\nПопулярные начинки:")
for result in results4["results"]["bindings"]:
    print(f"{result['topping']['value']}: {result['count']['value']}")

query5 = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX pizza: <http://www.co-ode.org/ontologies/pizza/pizza.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX ex: <http://example.org/vegetarian#>

CONSTRUCT {
	?pizza ex:isVegetarian true .
	?pizza ex:hasTopping ?topping .
}
WHERE {
  	{
    	?pizza owl:equivalentClass ?Class .
    	?Class owl:intersectionOf ?intersec .
    	?intersec ?about pizza:Pizza .
    	?pizza rdfs:label ?name .
 	}
  	UNION
  	{
  		?pizza rdfs:subClassOf pizza:NamedPizza .
    	?pizza rdfs:label ?name .
  	}
  	# Получение начинок
  	{
  		?pizza owl:equivalentClass ?Class .
    	?Class owl:intersectionOf ?intersec .
  		?intersec rdf:rest*/rdf:first ?Restriction .
  		?Restriction owl:someValuesFrom ?topping .
  	}
  	UNION
  	{
  		?pizza rdfs:subClassOf ?Restriction .
  		?Restriction owl:someValuesFrom ?topping .
  	}
    FILTER (?topping != pizza:MeatTopping)
}
"""

# Для CONSTRUCT запросов меняем формат вывода
sparql.setReturnFormat(XML)
results5 = run_query(query5)
print("CONSTRUCT запрос выполнен")

# Сохранение результатов
results5.serialize("vegetarian_pizzas.rdf", format="xml")

from rdflib import Graph, Namespace
from rdflib.plugins.stores import sparqlstore

# Создание графа с SPARQL endpoint
store = sparqlstore.SPARQLUpdateStore()
store.open(('http://localhost:3030/pizza-ds/sparql', 'http://localhost:3030/pizza-ds/update'))
g = Graph(store)
# Определение namespace
PIZZA = Namespace("http://www.co-ode.org/ontologies/pizza/pizza.owl#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

# Запрос через RDFLib
query6 = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?pizza ?name
WHERE {
  	{
    	?pizza owl:equivalentClass ?Class .
    	?Class owl:intersectionOf ?intersec .
    	?intersec ?about pizza:Pizza .
  		?pizza rdfs:label ?name .
 	}
  	UNION
  	{
  		?pizza rdfs:subClassOf pizza:NamedPizza .
    	?pizza rdfs:label ?name .
  	}
}
LIMIT 5
"""

results6 = g.query(query6, initNs={"pizza": PIZZA, "rdfs": RDFS})
print("\nРезультаты через RDFLib:")
for row in results6:
    print(f"{row.pizza} - {row.name}")

import xml.etree.ElementTree as ET
#from xml.dom import minidom

def generate_ontology_report():
    queries = {
        "total_classes": """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT (COUNT(DISTINCT ?class) AS ?count)
            WHERE { ?class a owl:Class }
        """,
        "total_properties": """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT (COUNT(DISTINCT ?prop) AS ?count)
            WHERE { ?prop a owl:ObjectProperty }
        """,
        "total_individuals": """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT (COUNT(DISTINCT ?ind) AS ?count)
            WHERE { ?ind a owl:NamedIndividual }
        """
    }
    
    report = {}
    for name, query in queries.items():
        results = ET.fromstring(run_query(query).toxml())
        if results:
            count = results[1][0][0][0].text
            report[name] = count

    # Сохранение отчета
    df = pd.DataFrame([report])
    df.to_csv("ontology_report.csv", index=False)
    return report

ontology_stats = generate_ontology_report()
print("\nСтатистика онтологии:")
for key, value in ontology_stats.items():
    print(f"{key}: {value}")

def test_endpoints():
    endpoints = [
        "http://localhost:3030/pizza-ds/sparql",
        "http://dbpedia.org/sparql",
    "http://query.wikidata.org/sparql"
    ]
    test_query = "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o } LIMIT 1"
    
    for endpoint in endpoints:
        try:
            sparql = SPARQLWrapper(endpoint)
            sparql.setQuery(test_query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            print(f"{endpoint}: Работает ({results['results']['bindings'][0]['count']['value']} triplets)")
        except:
            print(f"{endpoint}: Не доступен")

test_endpoints()