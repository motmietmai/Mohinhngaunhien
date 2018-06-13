'''
Created on Jul 18, 2011

@author: kjell
'''
from api.base_classifier import BaseClassifier
from api.specialized_hmm import SpecializedHMM
from api import simple_image_feature_extractor
from api.simple_image_feature_extractor import SimpleImageFeatureExtractor
from java.io import File
import shutil
import unittest

class CharacterClassifier(BaseClassifier):
    '''
    Works as baseClassifier with some extra features for character classification
    '''


    def __init__(self,
                 characters_with_examples=None,
                 nr_of_hmms_to_try=3,
                 fraction_of_examples_for_test=0.1,
                 train_with_examples=True,
                 initialisation_method=SpecializedHMM.InitMethod.count_based,
                 feature_extractor=None):
        '''
        See baseClassifier
        '''
        # if from_string_string != None:
        #     #init from string
        #     #"\n\n"+ in the next row is for jython bug 1469
        #     feature_extractor_parameters,classifer_string = eval("\n\n"+from_string_string)
        #     nr_of_divisions,size_classification_factor = feature_extractor_parameters
        #     self.feature_extractor = SimpleImageFeatureExtractor(nr_of_divisions, 
        #                                                          size_classification_factor)
        #     self.nr_of_segments = nr_of_divisions
        #     super(CharacterClassifier,self).__init__(from_string_string=classifer_string)
        #     return
        #Feature extractor is voluntary but is necessary if the classify_image
        #method shall be used
        self.feature_extractor = feature_extractor
        #Get the number of segments created by the feature extractor
        #by looking at the length of a training example
        label,examples = characters_with_examples[0]
        self.nr_of_segments = len(examples[0])
        
        new_characters_with_examples = []

        # new_characters_with_examples[0] = ('AAA...AAA', ['hfhhhhhiiii', ..., 'hfihhhhdfdi']):11 chu A
        for label,examples in characters_with_examples:
            new_characters_with_examples.append((label*self.nr_of_segments,examples))

        super(CharacterClassifier,self).__init__(new_characters_with_examples,
                                                 nr_of_hmms_to_try,
                                                 fraction_of_examples_for_test,
                                                 train_with_examples,
                                                 initialisation_method,
                                                 alphabet=SimpleImageFeatureExtractor.feature_ids)
    
    def classify_character_string(self,string):
        classification = super(CharacterClassifier, self).classify(string)
        return classification[0]
    
    def classify_image(self,buffered_image):
        string = self.feature_extractor.extract_feature_string(buffered_image)
        return self.classify_character_string(string)
    
    def test(self,test_examples):
        '''
        See baseClassifier.test()
        '''
        new_test_examples = []
        for label, examples in test_examples:
            new_test_examples.append((label * self.nr_of_segments, examples))
        return super(CharacterClassifier, self).test(new_test_examples)
    
    def to_string(self):
        if self.feature_extractor == None:
            raise "feature_extractor must be given if the character classifier shall be stringified"
        else:    
            feature_extractor_parameters = (self.feature_extractor.nr_of_divisions,
                                            self.feature_extractor.size_classification_factor)
        base_classifier_string = super(CharacterClassifier,self).to_string()
        return str((feature_extractor_parameters,
                    base_classifier_string))
        
        


if __name__ == "__main__":
    
    unittest.main()
        