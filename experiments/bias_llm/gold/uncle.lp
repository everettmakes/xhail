% Gold: uncle — chain through an intermediate person (needs a -person output link).
% target: uncle(X,Y) :- brother(X,Z), parent(Z,Y).
person(ann). person(ben). person(cal). person(dee). person(eli). person(fay). person(gus).
brother(ben,ann). brother(cal,dee). brother(gus,fay).
parent(ann,eli). parent(dee,fay). parent(fay,gus).
#example uncle(ben,eli).
#example uncle(cal,fay).
#example not uncle(ann,eli).
#example not uncle(ben,fay).
#example not uncle(gus,eli).
#example not uncle(dee,fay).
#modeh uncle(+person,+person).
#modeb brother(+person,-person).
#modeb parent(+person,+person).
