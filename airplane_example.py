import cv2
import numpy as np
import time
import copy
import random

class Environment:
	def __init__(self, map):
		self.map = map

	def getDistance(self, x, y, sensor):
		d = np.where(self.map[:,int(x),0] == 0)[0][0] - y

		if sensor:
			d += random.gauss(0.0, 1.0)

		return d

class Airplane:
	def __init__(self, x, y, v):
		self.x = x
		self.y = y
		self.v = v

	def update(self, dt):
		self.x += self.v * dt + random.gauss(0.0, 0.01)

class Drawer:
	def __init__(self):
		pass

	def drawPlane(self, img, x, y, d):
		cv2.rectangle(img, (int(x-20),int(y-2)), (int(x),int(y+2)), (255,0,255), 10)
		cv2.line(img, (int(x),int(y)), (int(x-10),int(y-20)), (255,0,255), 5)
		cv2.line(img, (int(x),int(y)), (int(x-10),int(y+20)), (255,0,255), 5)

		for i in range(int(d)):
			if i % 8 == 0:
				cv2.circle(img, (int(x),int(y+i)), 2, (0,0,255), -1)

		return img

	def drawParticle(self, img, particles, color):
		for particle in particles:
			cv2.circle(img, (int(particle.x),int(50)), 5, color, -1)

		return img

class Particle:
	def __init__(self, x, y, w):
		self.x = x
		self.y = y
		self.w = w

	def update(self, v, dt):
		self.x += v * dt + random.gauss(0.0, 0.5)
		self.x = np.clip(self.x,0,800-1)

def calcGaussianProbability(mu, sigma, x):
	return np.exp(-((mu - x) ** 2) / (sigma**2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))

if __name__=='__main__':
	img = cv2.imread('./mountain.png')
	height, width, _ = img.shape

	env = Environment(map=img)
	drawer = Drawer()
	airplane = Airplane(x=100, y=100, v=10)

	particles = []
	resampling_particels = []

	start, end = time.time(), time.time()
	while True:
		start = time.time()

		# make initial particles
		if len(particles) == 0:
			sampling = 1000
			for i in range(sampling):
				ranx = np.random.randint(width)
				particle = Particle(x=ranx, y=airplane.y, w=0)
				particles.append(particle)

		dt = start - end
		# dt = 0.01

		steps = 1
		for _ in range(steps):
			airplane.update(dt)
			measure_d = env.getDistance(airplane.x, airplane.y, True)

			for particle in particles:
				particle.update(airplane.v, dt)

		# calculate the weight of each particle
		w = []
		for particle in particles:
			d = env.getDistance(particle.x, particle.y, False)
			prob = calcGaussianProbability(d, 100.0, measure_d)
			particle.w = prob
			w.append(prob)

		# resampling
		resampling_particels = []

		index = int(np.random.random()*sampling)
		beta = 0.0
		mw = max(w)

		for i in range(sampling):
			beta += np.random.random() * 2.0 * mw

			while beta > w[index]:
				beta -= w[index]
				index = (index + 1) % sampling

			resampling_particels.append(copy.copy(particles[index]))

		# particles update
		particles = resampling_particels

		end = time.time()

		frame = drawer.drawPlane(copy.copy(img), airplane.x, airplane.y, measure_d)
		drawer.drawParticle(frame, particles, (255,0,0))
		drawer.drawParticle(frame, resampling_particels, (0,255,0))

		cv2.imshow('frame', frame)

		if cv2.waitKey(10) & 0xff == ord('q'):
			break
