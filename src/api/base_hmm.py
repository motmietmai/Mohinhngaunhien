'''
Created on Jul 7, 2011

@author: kjellw
'''

from hmm import HMM
from specialized_hmm import SpecializedHMM
from specialized_hmm import zeros
from specialized_hmm import random_list_with_sum
from specialized_hmm import fill_list_with_zeros_in_beginning_to_size
from specialized_hmm import zeros_and_random_with_sum1
from specialized_hmm import list_with_sum_and_equal_elements
from base_examples_generator import get_example_alphabet
from base_examples_generator import generate_examples_for_base
from random import random
import unittest
from specialized_hmm import SpecializedHMM


class baseHMM(SpecializedHMM):
    '''
    A HMM that represent a sequence of letter that form a base.
    It is implemented in the way described in the paper:
    Initial model selection for the Baum-Welch algorithm as applied to 
    HMMs of DNA sequences.
    '''

    def __init__(self, 
                 base_length=7, 
                 init_method=SpecializedHMM.InitMethod.random, 
                 training_examples=[],
                 alphabet=get_example_alphabet()):
        '''
        Training examples is only used if InitMethod.count_based is used
        '''
        # if from_string_string != None:
        #     #init from string
        #     pi, A, B, V = eval(from_string_string)
        #     super(baseHMM,self).__init__(pi, A, B, V)
        #     return
        self.base_length = base_length
        self.init_method = init_method
        self.training_examples = training_examples
        self.alphabet = alphabet
        if(self.init_method==SpecializedHMM.InitMethod.count_based and
           len(self.training_examples)==0):
            raise "Training examples needs to be provided when init method is count based"
        
        #Construct the state transition matrix
        self.number_of_states = base_length + 2
        
        #state transition matrix
        A = []
        #From state 1 to state 2 the probability is 
        state1 = zeros(self.number_of_states)
        state1[1]=1
        #state1 = [0, 1, 0, 0, .. ,0]

        A.append(state1)
        for i in range(1,self.number_of_states-1):
            state_row = self.init_transition_matrix_row(i)
            A.append(state_row)
        #last state can only be transfered to state1 with probability 1
        last_state = zeros(self.number_of_states)
        last_state[0]=1
        A.append(last_state)
        #init state emission probabilities...
        self.number_of_emissions = len(self.alphabet) + 2
        B = []
        #init the first row with specific probability for @
        B.append(zeros(self.number_of_emissions))
        B[0][0] = 1
        #init the rest emission probabilities without the last row
        for i in range(1, self.number_of_states-1):
            B.append(self.init_emission_probablity_matrix_row(i))
        #init the last row for specific probability for $
        B.append(zeros(self.number_of_emissions))
        B[self.number_of_states-1][self.number_of_emissions-1] = 1
        #Set of emission symbols
        V = ['@'] + self.alphabet + ['$']
        #Initial state
        pi = zeros(self.number_of_states)
        pi[0] = 1
        
        super(baseHMM,self).__init__(pi, A, B, V)

    def init_transition_matrix_row(self, row_index):
        if(self.init_method==SpecializedHMM.InitMethod.random):
            return zeros_and_random_with_sum1(self.number_of_states, self.number_of_states-row_index)
        elif(self.init_method==SpecializedHMM.InitMethod.count_based):
            row = (zeros(row_index) + 
                   list_with_sum_and_equal_elements(self.number_of_states-row_index-1,0.2))
            row.insert(row_index+1,0.8)
            return row            
        else:
            raise "Init Method Not Supported"
    
    def init_emission_probablity_matrix_row(self, row_index):
        if(self.init_method==SpecializedHMM.InitMethod.random):
            row = [0] + random_list_with_sum(self.number_of_emissions-2, 1) + [0]
            return row
        elif(self.init_method==SpecializedHMM.InitMethod.count_based):
            nr_of_training_examples = len(self.training_examples)
            alphabet = self.alphabet
            alphabet_size = len(alphabet)
            def count_position(position):
                #pseudocount
                use_pseudocount = True
                uniform_pseudocount = False
                if use_pseudocount:
                    if uniform_pseudocount:
                        count_list = zeros(alphabet_size)
                        for i in range(0,alphabet_size):
                            count_list[i]= (nr_of_training_examples*0.1)/alphabet_size
                    else:
                        count_list = random_list_with_sum(alphabet_size,
                                                          nr_of_training_examples*0.1)
                else:
                    count_list = zeros(alphabet_size)
                
                #Do the counting
                for e in self.training_examples:
                    if position < len(e):
                        character_index = alphabet.index(e[position])
                        count_list[character_index] = count_list[character_index] + 1
                return count_list
            count_list = count_position(row_index-1)
          
            total_count = sum(count_list)
            def normalize_element(element):
                return element / total_count
            row = map(normalize_element, count_list)
            return [0] + row + [0]
        else:
            raise "Init Method Not Supported"
    
    def observation_from_base(self,base):
        base_with_special_start_and_end = "@" +  base +  "$"
        observation_list = []
        for letter in base_with_special_start_and_end:
            observation_list.append(self.V.index(letter))
        
        return observation_list
    
    
    def train_baum_welch(self, training_examples):
        observation_list = []
        for base in training_examples:
            observation_list = observation_list + self.observation_from_base(base)
        self.baum_welch(observation_list)
        
    # def train_baum_welch_bakis(self, training_examples):
    #     '''bakis does not seem to work, see autotest'''
    #     observation_list = []
    #     for base in training_examples:
    #         observation_list.append(self.observation_from_base(base))
    #     self.baum_welch_bakis(observation_list)
        
    def train_until_stop_condition_reached(self, 
                                           training_examples, 
                                           delta = 0.003, 
                                           test_examples=None,
                                           max_nr_of_iterations=100):
        ''' Train the model using Baum Welch until stop condition is met.
            stop condition improvement < delta
           
            Parameters:
            training_examples - the example bases to train with
            delta - see stop condition
            test_examples the examples used to test for improvement the training examples are used if
            set to default None'''
        
        actual_test_examples = []
        if test_examples==None:
            actual_test_examples = training_examples
        else:
            actual_test_examples = test_examples
        
        score = 0
        old_score = -1
        improvement = score - old_score
        iter = 0
        while improvement > delta and iter < max_nr_of_iterations:
            iter = iter + 1
            self.train_baum_welch(training_examples)
            old_score = score
            score = self.test(actual_test_examples)
            improvement = score - old_score
    
    def test(self, base_list):
        '''Returns the likelihood of the base given the model'''
        probabilities_for_bases = []
        for base in base_list:
            O = self.observation_from_base(base)
            #alpha_matrix = self.calc_forward(O)
            #last_row = alpha_matrix[len(alpha_matrix)-1]
            probabilities_for_bases.append(self.probability_of_observation(O))
        average = sum(probabilities_for_bases)/len(probabilities_for_bases)
        return average         
        
if __name__ == "__main__":
   
    unittest.main()