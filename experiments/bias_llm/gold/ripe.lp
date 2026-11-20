% Gold: ripe fruit — the rule needs a ground constant (#hue), not a variable.
% target: ripe(F) :- colour(F,red).
fruit(f1). fruit(f2). fruit(f3). fruit(f4). fruit(f5). fruit(f6).
hue(red). hue(green). hue(yellow).
colour(f1,red). colour(f2,green). colour(f3,red). colour(f4,yellow). colour(f5,green). colour(f6,red).
#example ripe(f1).
#example ripe(f3).
#example not ripe(f2).
#example not ripe(f4).
#modeh ripe(+fruit).
#modeb colour(+fruit,#hue).
