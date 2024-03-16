# dict storing rules for combining two IUPAC letters
IUPAC_TO_IUPAC = {
        "A" : {"A":"A", "C":"M", "G":"R", "T":"W", "M":"M", "R":"R", "W":"W", 
               "S":"V", "Y":"H", "K":"D", "V":"V", "H":"H", "D":"D", "B":"N", 
               "N":"N", " ":"A",},
        "C" : {"A":"M", "C":"C", "G":"S", "T":"Y", "M":"M", "R":"V", "W":"H", 
               "S":"S", "Y":"Y", "K":"B", "V":"V", "H":"H", "D":"N", "B":"B", 
               "N":"N", " ":"C",},
        "G" : {"A":"R", "C":"S", "G":"G", "T":"K", "M":"V", "R":"R", "W":"D", 
               "S":"S", "Y":"B", "K":"K", "V":"V", "H":"N", "D":"D", "B":"B", 
               "N":"N", " ":"G",},
        "T" : {"A":"W", "C":"Y", "G":"K", "T":"T", "M":"H", "R":"D", "W":"W", 
               "S":"B", "Y":"Y", "K":"K", "V":"N", "H":"H", "D":"D", "B":"B", 
               "N":"N", " ":"T",},
        "M" : {"A":"M", "C":"M", "G":"V", "T":"H", "M":"M", "R":"V", "W":"H", 
               "S":"V", "Y":"H", "K":"N", "V":"V", "H":"H", "D":"N", "B":"N", 
               "N":"N", " ":"M",},
        "R" : {"A":"R", "C":"V", "G":"R", "T":"D", "M":"V", "R":"R", "W":"D", 
               "S":"V", "Y":"N", "K":"D", "V":"V", "H":"N", "D":"D", "B":"N", 
               "N":"N", " ":"R",},
        "W" : {"A":"W", "C":"H", "G":"D", "T":"W", "M":"H", "R":"D", "W":"W", 
               "S":"N", "Y":"H", "K":"D", "V":"N", "H":"H", "D":"D", "B":"N", 
               "N":"N", " ":"W",},
        "S" : {"A":"V", "C":"S", "G":"S", "T":"B", "M":"V", "R":"V", "W":"N", 
               "S":"S", "Y":"B", "K":"B", "V":"V", "H":"N", "D":"N", "B":"B", 
               "N":"N", " ":"S",},
        "Y" : {"A":"H", "C":"Y", "G":"B", "T":"Y", "M":"H", "R":"N", "W":"H", 
               "S":"B", "Y":"Y", "K":"B", "V":"N", "H":"H", "D":"N", "B":"B", 
               "N":"N", " ":"Y",},
        "K" : {"A":"D", "C":"B", "G":"K", "T":"K", "M":"N", "R":"D", "W":"D", 
               "S":"B", "Y":"B", "K":"K", "V":"N", "H":"N", "D":"D", "B":"B", 
               "N":"N", " ":"K",},
        "V" : {"A":"V", "C":"V", "G":"V", "T":"N", "M":"V", "R":"V", "W":"N", 
               "S":"V", "Y":"N", "K":"N", "V":"V", "H":"N", "D":"N", "B":"N", 
               "N":"N", " ":"V",},
        "H" : {"A":"H", "C":"H", "G":"N", "T":"H", "M":"H", "R":"N", "W":"H", 
               "S":"N", "Y":"H", "K":"N", "V":"N", "H":"H", "D":"N", "B":"N", 
               "N":"N", " ":"H",},
        "D" : {"A":"D", "C":"N", "G":"D", "T":"D", "M":"N", "R":"D", "W":"D", 
               "S":"N", "Y":"N", "K":"D", "V":"N", "H":"N", "D":"D", "B":"N", 
               "N":"N", " ":"D",},
        "B" : {"A":"N", "C":"B", "G":"B", "T":"B", "M":"N", "R":"N", "W":"N", 
               "S":"B", "Y":"B", "K":"B", "V":"N", "H":"N", "D":"N", "B":"B", 
               "N":"N", " ":"B",},
        "N" : {"A":"N", "C":"N", "G":"N", "T":"N", "M":"N", "R":"N", "W":"N", 
               "S":"N", "Y":"N", "K":"N", "V":"N", "H":"N", "D":"N", "B":"N", 
               "N":"N", " ":"N",},
        " " : {"A":"A", "C":"C", "G":"G", "T":"T", "M":"M", "R":"R", "W":"W", 
               "S":"S", "Y":"Y", "K":"K", "V":"V", "H":"H", "D":"D", "B":"B", 
               "N":"N", " ":" ",}
    }

IUPAC_TO_LIST = {
    "A" : ["A"],
    "C" : ["C"],
    "G" : ["G"],
    "T" : ["T"],
    "M" : ["A", "C"],
    "R" : ["A", "G"],
    "W" : ["A", "T"],
    "S" : ["C", "G"],
    "Y" : ["C", "T"],
    "K" : ["G", "T"],
    "V" : ["A", "C", "G"],
    "H" : ["A", "C", "T"],
    "D" : ["A", "G", "T"],
    "B" : ["C", "G", "T"],
    "N" : ["A", "C", "G", "T"],
}

IUPAC_NOT = {
    "A" : "B",
    "C" : "D",
    "G" : "H",
    "T" : "V",
    "M" : "K",
    "R" : "Y",
    "W" : "S",
    "S" : "W",
    "Y" : "R",
    "K" : "M",
    "V" : "T",
    "H" : "G",
    "D" : "C",
    "B" : "A",
}