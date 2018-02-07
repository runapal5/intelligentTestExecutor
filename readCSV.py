import csv
from nltk.tokenize import word_tokenize, word_tokenize
#----------------------------------------------------------------------
def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    for row in reader:
       # print(" ".join(row))
        print(row[1])
        print(word_tokenize(row[1]))
#----------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "Req_BM.csv"
    with open(csv_path, "rb") as f_obj:
        csv_reader(f_obj)
