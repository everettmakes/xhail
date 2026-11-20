% Gold: sprinklers — Event Calculus; learn which event causes the observed fluent changes.
% target: happens(spray(kitchen),T) :- holdsAt(smoke(kitchen),T).
% target: happens(spray(hall),T) :- holdsAt(smoke(hall),T).
holdsAt(F,T) :- time(T), time(S), S<T, happens(E,S), initiates(E,F,S), not clipped(S,F,T).
clipped(S,F,T) :- time(S), time(T), time(R), S<R, R<T, happens(E,R), terminates(E,F,R).
holdsAt(F,T) :- time(T), initially(F), not clipped(0,F,T).
time(0). time(1). time(2). time(3). time(4).
room(kitchen). room(hall).
initiates(ignite(R), smoke(R), T) :- room(R), time(T).
terminates(spray(R), smoke(R), T) :- room(R), time(T).
initiates(spray(R), wet(R), T) :- room(R), time(T).
happens(ignite(kitchen), 0).
happens(ignite(hall), 2).
#example holdsAt(wet(kitchen), 2).
#example holdsAt(wet(hall), 4).
#example not holdsAt(smoke(kitchen), 2).
#example not holdsAt(wet(hall), 3).
#example not holdsAt(wet(kitchen), 1).
#modeh happens(spray(#room), +time).
#modeb holdsAt(smoke(#room), +time).
