from docx import Document
import numpy as np
import pandas as pd
import os

document_list = []

document_list = [x for x in os.listdir(".") if x.endswith(".docx")]

documentText_list = []

for documentName in document_list:
    document = Document(documentName)
    documentText = ''
    try:
        for para in document.paragraphs:
            # print para.text
            documentText = documentText + para.text            
    except:
           print "Exception in document ", documentName
    documentText_list.append(documentText)
    


documents_dict = {}


documents_dict['Title'] = document_list
documents_dict['Synopsis'] = documentText_list

df = pd.DataFrame.from_dict(documents_dict)
#print df

print df['Title'].tolist()

movie_titles = documents_dict['Title']
movie_synopses = documents_dict['Synopsis']


print 'Movie:', movie_titles[0]
print 'Movie Synopsis:', movie_synopses[0][:58]


