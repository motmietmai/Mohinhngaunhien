from api.base_hmm import baseHMM
from api.base_examples_generator import generate_examples_for_bases,\
    get_example_alphabet
from api.specialized_hmm import SpecializedHMM
import unittest
import math

class BaseClassifier(object):
    
    def __init__(self,
                 bases_with_examples=None,
                 nr_of_hmms_to_try=3,
                 fraction_of_examples_for_test=0.1,
                 train_with_examples=True,
                 initialisation_method=SpecializedHMM.InitMethod.count_based,
                 alphabet=get_example_alphabet()):
        '''
        Parameters:
        bases_with_examples - is a list of tuples were the first element in the tuples
        is a string representing a base that the classifier should handle and the second
        element is a list of training examples for that base.
        nr_of_hmms_to_try - creates nr_of_hmms_to_try hmms for each base and selects the one with
        highest probability for the test examples
        fraction_of_examples_for_test -  fraction of the training examples that will be used for test
        train_with_examples - if training should be performed. Otherwise init will be done but not training
        All training examples will be used for both test and training if it is set to 0
        '''
        # if from_string_string != None:
        #     #init from string
        #     #"\n\n"+ in the next row is for jython bug 1469
        #     bases,stringified_hmms = eval("\n\n"+from_string_string)
        #     def destringify_hmm(hmm_string):
        #         return baseHMM(from_string_string=hmm_string)
        #     hmms = map(destringify_hmm,stringified_hmms)
        #     self.hmms_for_bases = hmms
        #     self.bases = bases
        #     return
        self.bases_with_examples = bases_with_examples
        self.nr_of_hmms_to_try = nr_of_hmms_to_try
        self.fraction_of_examples_for_test = fraction_of_examples_for_test
        self.initialisation_method  = initialisation_method
        self.alphabet = alphabet
        self.train(train_with_examples)
        
    def train(self,train_with_examples=True):
        self.bases = []
        self.hmms_for_bases = []
        for base,training_examples in self.bases_with_examples:
          
            self.bases.append(base)
            test_examples = []
            actual_training_examples = []
            if(self.fraction_of_examples_for_test == 0):
                test_examples = training_examples
                actual_training_examples = training_examples
            else:
                change_pot_at = len(training_examples)*self.fraction_of_examples_for_test
                for i in range(len(training_examples)):
                    if(i<change_pot_at):
                        test_examples.append(training_examples[i])
                    else:
                        actual_training_examples.append(training_examples[i])
                
            base_hmm = self.create_hmm_for_base(base, 
                                                actual_training_examples,
                                                test_examples,
                                                self.nr_of_hmms_to_try,
                                                train_with_examples)
            self.hmms_for_bases.append(base_hmm)

    
    def create_hmm_for_base(self, 
                            base, 
                            training_examples, 
                            test_examples, 
                            nr_of_hmms_to_try,
                            train_with_examples):
        #Create nr_of_hmms_to_try hmms and select the one with the best result
        results=[]
        hmms=[]
        for i in range(nr_of_hmms_to_try):
            if(self.initialisation_method==SpecializedHMM.InitMethod.count_based):
                hmm = baseHMM(len(base), 
                              SpecializedHMM.InitMethod.count_based,
                              training_examples,
                              alphabet=self.alphabet)
            elif(self.initialisation_method==SpecializedHMM.InitMethod.random):
                hmm = baseHMM(len(base), 
                              SpecializedHMM.InitMethod.random,
                              alphabet=self.alphabet)
            else:
                raise "Init method not supported"
            if train_with_examples:
                try:
                    hmm.train_until_stop_condition_reached(training_examples, 0.0001, test_examples)
                except ZeroDivisionError:
                    print("Divide by zero while training")
            hmms.append(hmm)
            result = hmm.test(test_examples)
            results.append(result)
            
        max_result = max(results)
        

        return hmms[results.index(max_result)]
    
    
    def classify(self,string):
        scores = []
        for hmm in self.hmms_for_bases:
            score = hmm.test([string])
            scores.append(score)
        print(scores)
        max_score = max(scores)
        return self.bases[scores.index(max_score)]
    
    def test(self,test_examples):
        '''
        Parameter:
        test_examples - is a list of tuples were the first element in the tuples
        is a string representing a base that the classifier should handle and the second
        element is a list of test examples for that base.
        
        Returns:
        Fraction of correctly classified test examples
        '''
        correctly_classified_counter = 0.0
        wrongly__classified_counter = 0.0
        for base, examples in test_examples:
            for example in examples:
                result = self.classify(example)
                if result== base:
                    correctly_classified_counter = correctly_classified_counter + 1
                else:
                    wrongly__classified_counter = wrongly__classified_counter + 1
        total_nr_of_tests = correctly_classified_counter + wrongly__classified_counter
        score = correctly_classified_counter / total_nr_of_tests
        return score
    
    def to_string(self):
        def hmm_to_string(hmm):
            return hmm.to_string()
        stringified_hmms = map(hmm_to_string, self.hmms_for_bases)
        return str((self.bases,stringified_hmms))
     

if __name__ == "__main__":
    
    unittest.main()
        
            
        