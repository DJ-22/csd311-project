/*  Solver 4: rule-based Dots-and-Boxes player (CSD311).

    Facts asserted by solvers/rules/rules_player.py:
      board (once per board size): box(B), edge(E), side_of(E, B), dots(D)
      position (once per query):   drawn(E), first_player(P), to_move(P)
    Query: best_move(E, Rule).

    The rules are tried in priority order and the cut commits to the first one
    that applies (12.pdf). They mirror the Python backend rule for rule: every
    tie is broken by the lowest edge index, so both backends pick the same move.

    A position is `now`, or `after(Xs)`: the current position with the extra
    edges Xs drawn (used to look at the board one move ahead).
*/

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(aggregate)).

:- dynamic box/1, edge/1, side_of/2, dots/1, drawn/1, first_player/1, to_move/1.


/* board relations*/

is_drawn(now, E)       :- drawn(E).
is_drawn(after(Xs), E) :- ( drawn(E) ; member(E, Xs) ).

undrawn(P, E) :- edge(E), \+ is_drawn(P, E).

sides_drawn(P, B, N) :-
    box(B),
    aggregate_all(count, ( side_of(E, B), once(is_drawn(P, E)) ), N).

% across(E, B, X): X is the box on the other side of edge E from box B, or none.
across(E, B, X) :- ( side_of(E, X0), X0 \== B -> X = X0 ; X = none ).

% missing(P, B, E): E is an undrawn side of box B.
missing(P, B, E) :- side_of(E, B), \+ is_drawn(P, E).

capturable(P, B) :- sides_drawn(P, B, 3).

% A safe move gives no box its third side.
safe(P, E) :- undrawn(P, E), \+ ( side_of(E, B), sides_drawn(P, B, 2) ).

mover_is_first :- to_move(P), first_player(P).


/* chains and loops (recursive, cf. above/2 in 12.pdf) */

two(P, B) :- sides_drawn(P, B, 2).

linked(P, B1, B2) :-
    side_of(E, B1), \+ is_drawn(P, E),
    side_of(E, B2), B2 \== B1, two(P, B2).

in_chain(_, B, B, _).
in_chain(P, B1, B2, Seen) :-
    linked(P, B1, X), \+ memberchk(X, Seen),
    in_chain(P, X, B2, [X|Seen]).

component(P, B, Boxes) :- findall(X, in_chain(P, B, X, [B]), Xs), sort(Xs, Boxes).

is_loop(P, Boxes) :- forall(member(B, Boxes), aggregate_all(count, linked(P, B, _), 2)).

% Chain ends that lead into a 3-sided box (always 0 for loops).
open_ends(P, Boxes, O) :-
    aggregate_all(count,
                  ( member(B, Boxes), side_of(E, B), \+ is_drawn(P, E),
                    across(E, B, X), X \== none, sides_drawn(P, X, 3) ),
                  O).

% components(P, Comps): every chain and loop, as comp(Kind, Length, OpenEnds, Boxes).
components(P, Comps) :-
    findall(B, two(P, B), Twos),
    collect_components(P, Twos, [], Comps).

collect_components(_, [], _, []).
collect_components(P, [B|Bs], Seen, Comps) :-
    (   memberchk(B, Seen)
    ->  collect_components(P, Bs, Seen, Comps)
    ;   component(P, B, Boxes),
        length(Boxes, L),
        (   is_loop(P, Boxes) -> Kind = loop, O = 0
        ;   Kind = chain, open_ends(P, Boxes, O)
        ),
        append(Boxes, Seen, Seen1),
        Comps = [comp(Kind, L, O, Boxes)|Rest],
        collect_components(P, Bs, Seen1, Rest)
    ).


/* capturable runs */

% A run starts at a 3-sided box and follows its missing side through 2-sided
% boxes: run(Boxes, 1) stops at the boundary or at a box with <= 1 side,
% run(Boxes, 2) reaches another 3-sided box (an opened loop).
run_from(A, run([A|Rest], Ends)) :-
    once(missing(now, A, E)), across(E, A, N),
    follow(N, E, Rest, Ends).

follow(none, _, [], 1) :- !.
follow(N, In, Boxes, Ends) :-
    sides_drawn(now, N, S),
    (   S =< 1 -> Boxes = [], Ends = 1
    ;   S =:= 3 -> Boxes = [N], Ends = 2
    ;   missing_other(N, In, Out), across(Out, N, Next),
        Boxes = [N|Rest],
        follow(Next, Out, Rest, Ends)
    ).

missing_other(N, In, Out) :- side_of(Out, N), Out \== In, \+ drawn(Out), !.

runs(Runs) :-
    findall(A, capturable(now, A), As),
    collect_runs(As, [], Runs).

collect_runs([], _, []).
collect_runs([A|As], Seen, Runs) :-
    (   memberchk(A, Seen)
    ->  collect_runs(As, Seen, Runs)
    ;   run_from(A, Run), Run = run(Boxes, _),
        append(Boxes, Seen, Seen1),
        Runs = [Run|Rest],
        collect_runs(As, Seen1, Rest)
    ).

% Capture moves of a run, ascending.
run_captures(run([A|_], 1), [E]) :- once(missing(now, A, E)).
run_captures(run(Boxes, 2), Cs) :-
    Boxes = [A|_], last(Boxes, Z),
    once(missing(now, A, E1)), once(missing(now, Z, E2)),
    sort([E1, E2], Cs).

% Boxes to hand back to keep control; fails when the run cannot be declined.
run_sacrifice(run(Boxes, 1), 2) :- length(Boxes, R), R >= 2.
run_sacrifice(run(Boxes, 2), 4) :- length(Boxes, R), R >= 4.

dd_able(Run) :- run_sacrifice(Run, _).

unopened(comp(loop, _, _, _)).
unopened(comp(chain, _, 0, _)).

long_comp(comp(loop, _, _, _)).
long_comp(comp(chain, L, _, _)) :- L >= 3.

% Remaining boxes minus the sacrifices needed to keep control: 2 per long chain,
% 4 per loop, none on the last one (a long chain if any is left).
control_value(Rem, V) :-
    aggregate_all(sum(L), member(comp(_, L, _, _), Rem), Total),
    aggregate_all(count, ( member(comp(chain, L1, _, _), Rem), L1 >= 3 ), NC),
    aggregate_all(count, member(comp(loop, _, _, _), Rem), NL),
    (   NC > 0 -> Last = 2
    ;   NL > 0 -> Last = 4
    ;   Last = 0
    ),
    V is Total - (2 * NC + 4 * NL - Last).

% The decline: leave one domino (one-ended run) or two dominoes (two-ended run).
double_deal_edge(run([A, Y], 1), E) :-
    once(missing(now, A, In)),
    side_of(E, Y), E \== In, \+ drawn(E), !.
double_deal_edge(run([_, C1, C2, _], 2), E) :-
    side_of(E, C1), side_of(E, C2), \+ drawn(E), !.


/* rule 1: endgame capture with control  */

endgame_double_deal(E) :-
    runs(Runs),
    include(dd_able, Runs, DD), DD \== [],
    components(now, Comps),
    include(unopened, Comps, Rem),
    once(( member(C, Rem), long_comp(C) )),
    findall(key(S1, R1, K1)-Run,
            ( member(Run, DD), run_sacrifice(Run, S1),
              Run = run(Bs, _), length(Bs, R1), run_captures(Run, [K1|_]) ),
            Keyed),
    keysort(Keyed, [_-Target|_]),
    exclude(==(Target), Runs, Others),
    (   Others \== []
    ->  findall(X, ( member(O, Others), run_captures(O, Xs), member(X, Xs) ), AllX),
        min_list(AllX, E)
    ;   run_sacrifice(Target, S), Target = run(TB, _), length(TB, R),
        run_captures(Target, [C0|_]),
        control_value(Rem, V),
        (   ( R > S ; S > V )
        ->  E = C0
        ;   double_deal_edge(Target, E)
        )
    ).


/* rule 2: plain capture  */

capture_move(E) :- aggregate_all(min(M), ( capturable(now, B), missing(now, B, M) ), E).


/*  rules 3 and 4: safe moves */

% Long chain rule: the first player wants dots + long chains even.
parity_ok(X) :-
    components(after([X]), Comps),
    aggregate_all(count, ( member(comp(chain, L, _, _), Comps), L >= 3 ), LC),
    dots(D),
    Parity is (D + LC) mod 2,
    (   mover_is_first -> Parity =:= 0 ; Parity =:= 1 ).

parity_safe_move(E) :- safe(now, E), parity_ok(E), !.

safe_move(E) :- safe(now, E), !.


/*  rule 5: minimum sacrifice  */

kind_rank(comp(chain, _, _, _), 0).
kind_rank(comp(loop, _, _, _), 1).

% A 2-chain is opened only by its middle edge (hard-hearted handout).
opening_move(comp(Kind, L, _, Boxes), X) :-
    member(B, Boxes), side_of(X, B), \+ drawn(X),
    (   Kind == chain, L =:= 2
    ->  across(X, B, O), memberchk(O, Boxes)
    ;   true
    ).

% Capture closure: keep taking capturable boxes; captures commute.
closure(Xs, Final) :-
    (   capturable(after(Xs), B), once(missing(after(Xs), B, M))
    ->  closure([M|Xs], Final)
    ;   Final = Xs
    ).

completed(P, N) :- aggregate_all(count, sides_drawn(P, _, 4), N).

% Boxes the opponent can capture after X is drawn.
captured_after(X, N) :-
    closure([X], Xs),
    completed(after(Xs), N1), completed(after([X]), N0),
    N is N1 - N0.

min_sacrifice(E) :-
    components(now, Comps),
    findall(key(Cost, KR, X),
            ( member(C, Comps), opening_move(C, X), kind_rank(C, KR), captured_after(X, Cost) ),
            Keys),
    Keys \== [],
    msort(Keys, [key(_, _, E)|_]).


/* the rule base: first match wins */

best_move(E, endgame_double_deal) :- endgame_double_deal(E), !.
best_move(E, capture)             :- capture_move(E), !.
best_move(E, parity_safe_move)    :- parity_safe_move(E), !.
best_move(E, safe_move)           :- safe_move(E), !.
best_move(E, min_sacrifice)       :- min_sacrifice(E), !.
best_move(E, min_sacrifice)       :- undrawn(now, E), !.

best_move(E) :- best_move(E, _).
