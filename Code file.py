
import json, time
from elasticUserSearch import ElasticUserSearch
import nltk
from elasticUserSearch_dsl import analyzer, Index, Mapping, Text, Integer



# Creating Analyzer
body = {
  "settings": {
    "number_of_shards": 1, 
    "analysis": {
      "analyzer": {
        "token_and_fold": {
          "type": "custom",
          "tokenizer": "lowercase"
        }
        ,
    "similarity": {
      "tf_idf": {
        "type": "scripted",
        "script": {-
          "source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;"
        }
      }
    }
  },
  "mappings": {
    "plots":{
      "properties": {
        "Plot": {
          "type": "text",
          "similarity": "tf_idf",
          "fields" : {
              "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
              }
            }
        }
      }
    }
  },
  
        "ngram_and_stop": {
          "type": "custom",
          "tokenizer": "lowercase",
          "filter": [
            "ngram", 
            "stop"
          ]
        },
        "stem": {
          "type": "custom",
          "tokenizer": "lowercase",
          "filter": [
            "stemmer"  
          ]
        }
      }
    }
}


# TF_DIF Mapping
mapping = {
    "properties": {
        "Plot": {
            "type": "text",
            "similarity": "tf_idf"
        }
    }
}

# Index UserSearching
UserSearch1 = {
  "query": {
    "bool": {
      "must": {
        "match": {
          "Source": "NBA"
        }
      },
      "filter": {
          "published": 2o15
      }
    }
  }
}


UserSearch2 = {
  "query": {
    "match": {
      "title": "Survivor Upset Use of 'Eye of the Tiger' With Kim Davis"
    }
  }
}

UserSearch3 = {
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "title": "The Streaming Media Device Landscape"
          }
        }
      ]
    }
  },
      "must": [
        {
          "match": {
            "media-type": "News"
          }
        }
      ],
  "sort": [
    {
      "Release Year": {
        "order": "desc"
      }
    }
  ]
}

# Printing Data set
def outputPrint(rslt):
        for hit in rslt['hits']['hits']:
            print(hit)

es = Elasticsearch("http://localhost:9200")

# Importing data set for performing Operations
print("Importing data")
with open('1000_Sample.json', 'r') as json_file:
    data = json_file.read()


print("Sending data to ElasticUserSearch")
# Load the document into ElasticUserSearch
es.bulk(data, index, doc_type)
print("Data upload successful")


# Stemming
print("This is Stemming")
rslt = es.indices.analyze(body={
    "analyzer": "stem"}, index="news")
outputPrint(rslt, True)

# Tockenization
print("This secition is for tockenization and case folding ")
rslt = es.indices.analyze(body={
    "analyzer":"token_and_fold"}, index="movies")
outputPrint(rslt, True)

# Stop-word removal and ngram
print("Stop-Word Removal and N-Gram Tokeniser")
rslt = es.indices.analyze(body={
    "analyzer": "ngram_and_stop, "text": "The Streaming Media Device Landscape"}, index="news")
outputPrint(rslt, True)



# User Search 1
print("NBA")
rslt = es.UserSearch(body=UserSearch1, index="news")
outputPrint(rslt, False)

# User search 2
print("Survivor Upset Use of 'Eye of the Tiger' With Kim Davis")
rslt = es.UserSearch(body=UserSearch2, index="news")
outputPrint(rslt, False)

# User Search 3
print("The Streaming Media Device Landscape")
rslt = es.UserSearch(body=UserSearch3, index="news")
outputPrint(rslt, False)



# Named Entity Recognition

import spacy
import json

#English NER model
nlp = spacy.load('en_core_web_sm')

#Opening our dataset
with open('example.json', 'r') as f:
    data = json.load(f)

# Looping through data file
for item in data:
    doc = nlp(item['text'])
    #Taking named entities to dictionary
    item['entities'] = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

# New jason file
with open('NER.json', 'w') as f:
    json.dump(data, f)



