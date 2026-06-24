def decouper_arguments(appel: str) -> List[str]:
    courant, morceaux = [], []
    niveau = 0

    for c in appel:

        if c == "(":
            niveau += 1
            courant.append(c)

        elif c == ")":
            niveau = max(0, niveau -1)
            courant.append(c)
        
        elif niveau == 0 and c == ",":
            morceaux.append("".join(courant))
            courant = []
        
        else:
            courant.append(c)
    
    morceaux.append("".join(courant))

    return morceaux

#---------------#         #------------#         #---------#         #------#

import re

_RE_PREFIXE = re.compile(r"[^#]{1,20}#\s*") # au début, un bloc de 1 à 20 caractères sans hashtag, suivi d'un hashtag et d'espaces optionnels

def nettoyer_tags(tags_bruts: List[str]) -> List[str]:
    
    resultat = []
    vus = set()

    for tag in tags_bruts:
        sbst = _RE_PREFIXE.sub("", tag)
        sbst = sbst.strip() # supprime les espaces, tabulations, retour à la ligne

        cle = sbst.lower()
        if cle not in vus:
            resultat.append(sbst)
            vus.add(cle)

    return resultat

# print(nettoyer_tags(["Sport# Tennis", "Cuisine# Pâtes", " tennis "]))

#---------------#         #------------#         #---------#         #------#

import unicodedata

_RE_CARACTERES = re.compile(r"[^a-z0-9]+")

def slugifier(titre: str) -> str:

    titre = titre.lower()

    l_titre = "".join(
        c for c in unicodedata.normalize("NFD", titre)
        if unicodedata.category(c) != "Mn"
    )

    # on fait le traitement des accents avant car _RE_CARACTERES croit que les accents ne sont pas des lettres

    ss_titre = _RE_CARACTERES.sub("-", l_titre)


    l_a_titre = ss_titre.strip("-")

    return l_a_titre


# print(slugifier(" DÖnnées ! 2.15 "))

#---------------#         #------------#         #---------#         #------#

