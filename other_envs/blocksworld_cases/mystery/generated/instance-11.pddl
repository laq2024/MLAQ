(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c k g a l j e h i b)
(:init 
(harmony)
(planet c)
(planet k)
(planet g)
(planet a)
(planet l)
(planet j)
(planet e)
(planet h)
(planet i)
(planet b)
(province c)
(province k)
(province g)
(province a)
(province l)
(province j)
(province e)
(province h)
(province i)
(province b)
)
(:goal
(and
(craves c k)
(craves k g)
(craves g a)
(craves a l)
(craves l j)
(craves j e)
(craves e h)
(craves h i)
(craves i b)
)))