import nlpaug.augmenter.word as naw

text = ", et par la seule transposition des parties ; de mesme que des epingles amassées confusement en un tas piquent de tous costez , au lieu qu' étant jointes ensemble , elles paroissent douces , et polies ; ou demesme qu' on touche les poils de l' herisson quand ils sont couchez sans aucun sentiment de douleur, au lieu qu' estant dressez ils picquent sensiblement . Enfin pour dire quelque chose de plus familier , voyez une pomme quand elle se pourrit , et qu' elle a pourtant encore quelque partie saine , quelle diversité n' y a -t-il point dans la couleur , dans l' odeur , dans la saveur , dans la mollesse ,"


aug = naw.RandomWordAug(action='insert', lang='fra')
augmented_text = aug.augment(text)

print(augmented_text)