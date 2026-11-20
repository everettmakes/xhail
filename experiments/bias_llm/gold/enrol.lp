% Gold: course enrolment — output link from the head's second argument, then a two-input literal.
% target: can_take(S,C) :- prereq(C,P), passed(S,P).
student(sam). student(tia). student(uma).
course(c1). course(c2). course(c3). course(c4).
prereq(c2,c1). prereq(c3,c2). prereq(c4,c1).
passed(sam,c1). passed(tia,c1). passed(tia,c2). passed(uma,c3).
#example can_take(sam,c2).
#example can_take(tia,c3).
#example can_take(tia,c4).
#example not can_take(sam,c3).
#example not can_take(uma,c2).
#example not can_take(uma,c4).
#modeh can_take(+student,+course).
#modeb prereq(+course,-course).
#modeb passed(+student,+course).
