function buffon(m)
  hit = 0
  for l = 1:m
    mp = rand()
    phi = (rand() * pi) - pi / 2
    xrechts = mp + cos(phi)/2
    xlinks = mp - cos(phi)/2
    if xrechts >= 1 || xlinks <= 0
      hit += 1
    end
  end
  miss = m - hit
  piapprox = m / hit * 2
end

buffon(10000000)
