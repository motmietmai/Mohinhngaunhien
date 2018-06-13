'''
Created on Jul 3, 2011

@author: kjell
'''

from random import random
from random import choice
import unittest

example_alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def get_example_alphabet():
    return example_alphabet

def generate_examples_for_base(base="dog", number_of_examples=100, poelap=0.03, poelenl=0.7, powlap=0.1, polmap=0.01, alphabet=example_alphabet):
    '''
    Function that generate misspelled versions of a base given propabilities
    defined by the parameters.  
        
    Parameters:
    base = the base that the examples shall be generated for
    poelap = probability of extra letter at position
    poelenl = probability of extra letter equals neighbor letter
    powlap = probability of wrong letter at position
    polmap = probability of letter missing at position
    number_of_examples = the number of examples that shall be generated
    
    Returns:
    A list of size number_of_examples containing versions of the base
    '''
    #Help functions:
    def true_with_probability(probability):
        return random() <= probability
    
    def neighbors_at_position(base, position):
        base_length = len(base)
        if(position==0):
            return [base[0]]
        elif position < base_length:
            return [base[position-1], base[position]]
        else:
            return [base[base_length-1]]
    
    def actual_or_other(letter):
        if(true_with_probability(polmap)):#Letter missing at position
            return ""
        else:
            if(true_with_probability(powlap)):#Wrong letter at position
                return choice(alphabet)
            else:
                return letter        
    
    def generate_example_for_base_from_pos(base,start_at_position=0):
        if start_at_position > len(base):
            return ""
        else:
            end = start_at_position == len(base)
            char_at_pos = "" if end else actual_or_other(base[start_at_position])
            rest = generate_example_for_base_from_pos(base,start_at_position+1)
            if(true_with_probability(poelap)):#probability of extra letter 
                if(true_with_probability(poelenl)):#probability of extra letter equals to neighbor
                    neighbor = choice(neighbors_at_position(base, start_at_position))
                    return neighbor + char_at_pos + rest
                else:
                    extra_letter = choice(alphabet)
                    return extra_letter + char_at_pos + rest
            else:
                return char_at_pos + rest
        
    #Generate the examples
    examples = []
    for i in range(number_of_examples): #@UnusedVariable
        examples.append(generate_example_for_base_from_pos(base))
    return examples


default_base_list = ["dog","cat","pig","love","hate",
                     "scala","python","summer","winter","night",
                     "daydream","nightmare","animal","happiness","sadness",
                     "tennis","feminism","fascism","socialism","capitalism"]

def generate_examples_for_bases(bases=default_base_list, number_of_examples=100, poelap=0.03, poelenl=0.7, powlap=0.1, polmap=0.01, alphabet=example_alphabet):
    '''
    Generate tuples for all bases in the list bases of the format:
    (base, list of training examples for the bases)
    
    See generate_examples_for_base for description of the rest of the parameters
    '''
    base_training_example_tuples = []
    for base in bases:
        base_training_example_tuples.append((base,generate_examples_for_base(base, number_of_examples, poelap, poelenl, powlap, polmap, alphabet)))
    return base_training_example_tuples


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_base_']
    unittest.main()

        
            