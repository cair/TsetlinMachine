# Copyright (c) 2019 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements a multiclass version of the Tsetlin Machine from paper arXiv:1804.01508
# https://arxiv.org/abs/1804.01508

#cython: boundscheck=False, cdivision=True, initializedcheck=False, nonecheck=False

import numpy as np
cimport numpy as np
import random
from libc.stdlib cimport rand, RAND_MAX

########################################
### The Multiclass Tsetlin Machine #####
########################################

cdef class MultiClassTsetlinMachine:
	cdef int number_of_classes
	cdef int number_of_clauses
	cdef int number_of_features
	cdef float s
	cdef int number_of_states

	cdef int[:,:,:] ta_state

	cdef int[:] clause_count
	cdef int[:,:,:] clause_sign

	cdef int[:] clause_output

	cdef int[:] class_sum

	cdef int[:] feedback_to_clauses

	cdef int threshold

	cdef int boost_true_positive_feedback
	
	# Initialization of the Tsetlin Machine
	def __init__(self, number_of_classes, number_of_clauses, number_of_features, number_of_states, s, threshold, boost_true_positive_feedback = 0):
		cdef int[:] target_indexes
		cdef int c,i,j,m

		self.number_of_classes = number_of_classes
		self.number_of_clauses = number_of_clauses
		self.number_of_features = number_of_features
		self.number_of_states = number_of_states
		self.s = s
		self.threshold = threshold
		self.boost_true_positive_feedback = boost_true_positive_feedback

		# The state of each Tsetlin Automaton is stored here. The automata are randomly initialized to either 'number_of_states' or 'number_of_states' + 1.
		self.ta_state = np.random.choice([self.number_of_states, self.number_of_states+1], size=(self.number_of_clauses, self.number_of_features, 2)).astype(dtype=np.int32)

		# Data structures for keeping track of which clause refers to which class, and the sign of the clause
		self.clause_count = np.zeros((self.number_of_classes,), dtype=np.int32)
		self.clause_sign = np.zeros((self.number_of_classes, self.number_of_clauses/self.number_of_classes, 2), dtype=np.int32)
		
		# Data structures for intermediate calculations (clause output, summation of votes, and feedback to clauses)
		self.clause_output = np.zeros(shape=(self.number_of_clauses,), dtype=np.int32)
		self.class_sum = np.zeros(shape=(self.number_of_classes,), dtype=np.int32)
		self.feedback_to_clauses = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)

		# Set up the Tsetlin Machine structure
		for i in xrange(self.number_of_classes):
			for j in xrange(self.number_of_clauses / self.number_of_classes):
				self.clause_sign[i,self.clause_count[i],0] = i*(self.number_of_clauses/self.number_of_classes) + j
				if j % 2 == 0:
					self.clause_sign[i, self.clause_count[i], 1] = 1
				else:
					self.clause_sign[i, self.clause_count[i], 1] = -1

				self.clause_count[i] += 1

	# Calculate the output of each clause using the actions of each Tsetline Automaton.
	# Output is stored an internal output array.
	cdef void calculate_clause_output(self, int[:] X, int predict=0):
		cdef int j,k
		cdef int action_include, action_include_negated
		cdef int all_exclude

		for j in xrange(self.number_of_clauses):				
			self.clause_output[j] = 1
			all_exclude = 1
			for k in xrange(self.number_of_features):
				action_include = self.action(self.ta_state[j,k,0])
				action_include_negated = self.action(self.ta_state[j,k,1])

				if action_include == 1 or action_include_negated == 1:
					all_exclude = 0

				if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
					self.clause_output[j] = 0
					break

			if predict == 1 and all_exclude == 1:
				self.clause_output[j] = 0

	# Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine)
	cdef void sum_up_class_votes(self):
		cdef int target_class
		cdef int j

		for target_class in xrange(self.number_of_classes):
			self.class_sum[target_class] = 0

			for j in xrange(self.clause_count[target_class]):
				self.class_sum[target_class] += self.clause_output[self.clause_sign[target_class,j,0]]*self.clause_sign[target_class,j,1]
			
			if self.class_sum[target_class] > self.threshold:
				self.class_sum[target_class] = self.threshold
			elif self.class_sum[target_class] < -self.threshold:
				self.class_sum[target_class] = -self.threshold

	########################################
	### Predict Target Class for Input X ###
	########################################

	def predict(self, int[:] X):
		cdef int target_class
		cdef int max_class
		cdef float max_class_sum
		
		###############################
		### Calculate Clause Output ###
		###############################

		self.calculate_clause_output(X, predict=1)

		###########################
		### Sum up Clause Votes ###
		###########################

		self.sum_up_class_votes()

		##########################################
		### Identify Class with Largest Output ###
		##########################################
		
		max_class_sum = self.class_sum[0]
		max_class = 0
		for target_class in xrange(1, self.number_of_classes):				
			if max_class_sum < self.class_sum[target_class]:
				max_class_sum = self.class_sum[target_class]
				max_class = target_class
			
		return max_class

	# Translates automata state to action 
	cdef int action(self, int state):
		if state <= self.number_of_states:
			return 0
		else:
			return 1

	# Get the state of a specific automaton, indexed by clause, feature, and automaton type (include/include negated).
	def get_state(self, int clause, int feature, int automaton_type):
		return self.ta_state[clause,feature,automaton_type]

	############################################
	### Evaluate the Trained Tsetlin Machine ###
	############################################

	def evaluate(self, int[:,:] X, int[:] y, int number_of_examples):	
		cdef int l, j
		cdef int errors
		cdef int max_class
		cdef float max_class_sum
		cdef int[:] Xi

		Xi = np.zeros((self.number_of_features,), dtype=np.int32)

		errors = 0
		for l in xrange(number_of_examples):
			###############################
			### Calculate Clause Output ###
			###############################
			
			for j in xrange(self.number_of_features):
					Xi[j] = X[l,j]
			self.calculate_clause_output(Xi, predict=1)

			###########################
			### Sum up Clause Votes ###
			###########################

			self.sum_up_class_votes()

			##########################################
			### Identify Class with Largest Output ###
			##########################################

			max_class_sum = self.class_sum[0]
			max_class = 0
			for target_class in xrange(1, self.number_of_classes):				
				if max_class_sum < self.class_sum[target_class]:
					max_class_sum = self.class_sum[target_class]
					max_class = target_class
			
			if max_class != y[l]:
				errors += 1
		
		return 1.0 - 1.0 * errors / number_of_examples

	##########################################
	### Online Training of Tsetlin Machine ###
	##########################################

	# The Tsetlin Machine can be trained incrementally, one training example at a time.
	# Use this method directly for online and incremental training.

	cpdef void update(self, int[:] X, int target_class):
		cdef int i, j
		cdef int negative_target_class
		cdef int action_include, action_include_negated

		# Randomly pick one of the other classes, for pairwise learning of class output 
		negative_target_class = int(self.number_of_classes * 0.5*rand()/(RAND_MAX/2+1))
		while negative_target_class == target_class:
			negative_target_class = int(self.number_of_classes * 0.5*rand()/(RAND_MAX/2+1))

		###############################
		### Calculate Clause Output ###
		###############################

		self.calculate_clause_output(X)

		###########################
		### Sum up Clause Votes ###
		###########################

		self.sum_up_class_votes()

		#####################################
		### Calculate Feedback to Clauses ###
		#####################################

		# Initialize feedback to clauses
		for j in xrange(self.number_of_clauses):
			self.feedback_to_clauses[j] = 0

		# Calculate feedback to clauses
		for j in xrange(self.clause_count[target_class]):
			if 1.0*rand()/RAND_MAX > (1.0/(self.threshold*2))*(self.threshold - self.class_sum[target_class]):
				continue

			if self.clause_sign[target_class,j,1] > 0:
				# Type I Feedback				
				self.feedback_to_clauses[self.clause_sign[target_class,j,0]] += 1

			elif self.clause_sign[target_class,j,1] < 0:
				# Type II Feedback
				self.feedback_to_clauses[self.clause_sign[target_class,j,0]] -= 1

		for j in xrange(self.clause_count[negative_target_class]):
			if 1.0*rand()/RAND_MAX > (1.0/(self.threshold*2))*(self.threshold + self.class_sum[negative_target_class]):
				continue

			if self.clause_sign[negative_target_class,j,1] > 0:
				# Type II Feedback
				self.feedback_to_clauses[self.clause_sign[negative_target_class,j,0]] -= 1

			elif self.clause_sign[negative_target_class,j,1] < 0:
				# Type I Feedback
				self.feedback_to_clauses[self.clause_sign[negative_target_class,j,0]] += 1
		
		#################################
		### Train Invididual Automata ###
		#################################

		for j in xrange(self.number_of_clauses):
			if self.feedback_to_clauses[j] > 0:
				####################################################
				### Type I Feedback (Combats False Negatives) ###
				####################################################

				if self.clause_output[j] == 0:		
					for k in xrange(self.number_of_features):	
						if 1.0*rand()/RAND_MAX <= 1.0/self.s:								
							if self.ta_state[j,k,0] > 1:
								self.ta_state[j,k,0] -= 1
													
						if 1.0*rand()/RAND_MAX <= 1.0/self.s:
							if self.ta_state[j,k,1] > 1:
								self.ta_state[j,k,1] -= 1

				elif self.clause_output[j] == 1:					
					for k in xrange(self.number_of_features):
						if X[k] == 1:
							if self.boost_true_positive_feedback == 1 or 1.0*rand()/RAND_MAX <= (self.s-1)/self.s:
								if self.ta_state[j,k,0] < self.number_of_states*2:
									self.ta_state[j,k,0] += 1

							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,1] > 1:
									self.ta_state[j,k,1] -= 1

						elif X[k] == 0:
							if self.boost_true_positive_feedback == 1 or 1.0*rand()/RAND_MAX <= (self.s-1)/self.s:
								if self.ta_state[j,k,1] < self.number_of_states*2:
									self.ta_state[j,k,1] += 1

							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,0] > 1:
									self.ta_state[j,k,0] -= 1
			
			elif self.feedback_to_clauses[j] < 0:
				#####################################################
				### Type II Feedback (Combats False Positives) ###
				#####################################################
				if self.clause_output[j] == 1:
					for k in xrange(self.number_of_features):
						action_include = self.action(self.ta_state[j,k,0])
						action_include_negated = self.action(self.ta_state[j,k,1])

						if X[k] == 0:
							if action_include == 0 and self.ta_state[j,k,0] < self.number_of_states*2:
								self.ta_state[j,k,0] += 1
						elif X[k] == 1:
							if action_include_negated == 0 and self.ta_state[j,k,1] < self.number_of_states*2:
								self.ta_state[j,k,1] += 1

	##############################################
	### Batch Mode Training of Tsetlin Machine ###
	##############################################

	def fit(self, int[:,:] X, int[:] y, int number_of_examples, int epochs=100):
		cdef int i, j, epoch
		cdef int example_id
		cdef int[:] Xi
		cdef int target_class
		cdef long[:] random_index

		Xi = np.zeros((self.number_of_features,), dtype=np.int32)
	
		random_index = np.arange(number_of_examples)

		for epoch in xrange(epochs):			
			np.random.shuffle(random_index)

			for i in xrange(number_of_examples):
				example_id = random_index[i]
				target_class = y[example_id]

				for j in xrange(self.number_of_features):
					Xi[j] = X[example_id,j]
				self.update(Xi, target_class)
		return
			
