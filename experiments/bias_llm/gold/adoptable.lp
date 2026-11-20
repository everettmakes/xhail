% Gold: adoptable dogs — conjunction of a positive and a negated condition.
% target: adoptable(X) :- vaccinated(X), not aggressive(X).
dog(d1). dog(d2). dog(d3). dog(d4). dog(d5). dog(d6).
vaccinated(d1). vaccinated(d2). vaccinated(d4). vaccinated(d5).
aggressive(d2). aggressive(d3). aggressive(d6).
#example adoptable(d1).
#example adoptable(d4).
#example not adoptable(d2).
#example not adoptable(d3).
#example not adoptable(d6).
#modeh adoptable(+dog).
#modeb vaccinated(+dog).
#modeb aggressive(+dog).
#modeb not aggressive(+dog).
#modeb not vaccinated(+dog).
