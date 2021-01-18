obs.i <- m > 0
hist(m[obs.i])
sum(obs.i)
alea <- sample(m[obs.i])
m.alea <- m
m.alea[obs.i] <- alea
hist(m.alea[obs.i])
mean(abs(m.alea - m))
sum(abs(m.alea - m))/sum(obs.i)
sqrt(sum((m.alea - m)^2)/sum(obs.i))
