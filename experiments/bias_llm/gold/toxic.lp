% Gold: toxic molecules — part-of structure (molecule -> atom) plus an atom property.
% target: toxic(M) :- has_atom(M,A), chlorine(A).
molecule(m1). molecule(m2). molecule(m3). molecule(m4). molecule(m5).
has_atom(m1,a1). has_atom(m1,a2). has_atom(m2,a3). has_atom(m3,a4). has_atom(m3,a5).
has_atom(m4,a6). has_atom(m5,a7). has_atom(m5,a8).
atom(A) :- has_atom(_,A).
chlorine(a2). chlorine(a5). chlorine(a8).
carbon(a1). carbon(a3). carbon(a4). carbon(a6). carbon(a7).
#example toxic(m1).
#example toxic(m3).
#example not toxic(m2).
#example not toxic(m4).
#modeh toxic(+molecule).
#modeb has_atom(+molecule,-atom).
#modeb chlorine(+atom).
#modeb carbon(+atom).
