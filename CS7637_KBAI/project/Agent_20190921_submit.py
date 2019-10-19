# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy as np

# Notes:
# - All images may have 255 for the alpha value, i.e., images are RGB


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        #print(problem.name, problem.problemType, problem.problemSetName, \
              #problem.hasVisual, problem.hasVerbal)
        
        # The index is used to look-up the structure for a problemType.
        # For the values, the first tuple contains the 'question' images and 
        # the second tuple contains the possible 'answer' images.
        layout = {
                '2x2': (('A', 'B', 'C'), 
                        ('1', '2', '3', '4', '5', '6')), 
                '3x3': (('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'), 
                        ('1', '2', '3', '4', '5', '6', '7', '8'))
                }
        
        if problem.problemType == '2x2':
            question = layout[problem.problemType][0]
            answers = layout[problem.problemType][1]
            
            # Extra pixel information in original RGBA format not necessary 
            # and converting to BW 8-pixel reduces operations
            imageA = Image.open(problem.figures[question[0]].visualFilename)
            imageA = imageA.convert('L')
            imageB = Image.open(problem.figures[question[1]].visualFilename)
            imageB = imageB.convert('L')
            imageC = Image.open(problem.figures[question[2]].visualFilename)
            imageC = imageC.convert('L')
            
            pdiffAB = self.pixelDifference(imageA, imageB)
            #print('Pixel difference between A and B =', str(pdiffAB))
            pdiffAC = self.pixelDifference(imageA, imageC)
            #print('Pixel difference between A and C =', str(pdiffAC))
            
            hdiffAB = self.histDifference(imageA, imageB)
            #print('Histogram difference between A and B =', str(hdiffAB))
            hdiffAC = self.histDifference(imageA, imageC)
            #print('Histogram difference between A and C =', str(hdiffAC))
                        
            # Test all the 'basic' transformations and return the transformations
            # that successfully map A-B and A-C.
            xformsBasicAB = self.compareXformsBasic(imageA, imageB)
            #print('Indicative transformation(s) for A-B is/are:', xformsBasicAB)
            xformsBasicAC = self.compareXformsBasic(imageA, imageC)
            #print('Indicative transformation(s) for A-C is/are:', xformsBasicAC)
            
            # If they exists, apply the indicative transformations on C and B.
            # Then check if the resulting image is one of the answers.
            # If the result is not one of the answers, iterate through the list
            # of valid basic transformations.
            if xformsBasicAB:
                count = 0
                answerFromAB = False
                while count < len(xformsBasicAB) and (not answerFromAB):
                    imageDfromAB = self.applyXformBasic(imageC, xformsBasicAB[count])
                    answerFromAB = self.findAnswer(problem, imageDfromAB, answers)
                    #print('Answer from A-B transformation:', answerFromAB)
                    count += 1
            #else:
                #print('No basic indicative transformation found for A-B.')
            
            if xformsBasicAC:
                count = 0
                answerFromAC = False
                while count < len(xformsBasicAC) and not answerFromAC:
                    imageDfromAC = self.applyXformBasic(imageB, xformsBasicAC[count])
                    answerFromAC = self.findAnswer(problem, imageDfromAC, answers)
                    #print('Answer from A-C transformation:', answerFromAC)
                    count += 1
            #else:
                #print('No basic indicative transformation found for A-C.')
            
            # If the transformations applied to C and B both result in 
            # the same answer, return that image as the answer.
            # Both transformations were identified
            if xformsBasicAB and xformsBasicAC:
                # If both transformations resulted in the same answer, 
                # that answer is the unambiguous solution.
                if answerFromAB == answerFromAC:
                    #print('Final answer:', answerFromAB)
                    return int(answerFromAB)
                # If both transformations were identified, but one did not 
                # result in an answer, choose the one that did find an answer.
                elif answerFromAB and not answerFromAC:
                    #print('Final answer:', answerFromAB)
                    return int(answerFromAB)
                elif answerFromAC and not answerFromAB:
                    #print('Final answer:', answerFromAC)
                    return int(answerFromAC)
                # If both transformations found answers but they differed, 
                # choose the A-B transformation (arbitrary heirarchy based 
                # upon what I perceive to be the more apparent relationship).
                elif answerFromAB != answerFromAC:
                    #print('Final answer:', answerFromAB)
                    return int(answerFromAB)
                # Neither transformation resulted in an available answer
                else:
                    #print('No matching answer image found. No final answer.')
                    return -1
            # Only the A-B transformation was identified
            elif xformsBasicAB:
                if answerFromAB:
                    #print('Only A-B transformation exists and it produces an answer. Final answer:', answerFromAB)
                    return int(answerFromAB)
                else:
                    #print('No valid transformation. No final answer.')
                    return -1
            # Only the A-C transformation was identified
            elif xformsBasicAC:
                if answerFromAC:
                    #print('Only A-C transformation exists and it produces an answer. Only A-C transformation exists. Final answer:', answerFromAC)
                    return int(answerFromAC)
                else:
                    #print('No valid transformation. No final answer.')
                    return -1
            else:
                #print('No valid transformation. No final answer.')
                return -1
        
        # Anything not 2x2 not handled at this point
        else:
            return -1
            
        #self.writeData(problem)
        #return -1

    # This function writes the data from getdata() to .txt files
    def writeData(self, problem):
        for f in problem.figures:
            #print(problem.figures[f].visualFilename)
            image = Image.open(problem.figures[f].visualFilename)
            image = image.convert('L')
            #print(image.getbands())
            
            filename = problem.figures[f].name + '.txt'
            with open(filename, 'w') as imageData:
                data = list(image.getdata())
                for d in data:
                    imageData.write(str(d) + '\n')

    def histDifference(self, image1, image2):
        '''Calculates the Euclidean distance between two image histogram vectors. 
        
        Each histogram vector is sequence of the counts of pixels in an image.
        '''        
        
        h1 = np.array(image1.histogram())
        h2 = np.array(image2.histogram())
        
        #print('Calculating histogram difference...')
        diff = h1 - h2
        euclid = np.array( [np.sqrt(d**2) for d in diff] ).sum()
        #print(euclid)
        euclidNorm = round(euclid/(len(diff)*256),10)
        
        #print('Returning histogram difference...')
        return euclidNorm

    def pixelDifference(self, image1, image2):
        '''Calculates the Euclidean distance between two image vectors. 
        
        Each image vector is a complete sequence of pixel data.
        '''
        
        pixels1 = np.array( list(image1.getdata()) )
        pixels2 = np.array( list(image2.getdata()) )
        
        #print('Calculating pixel difference...')
        diff = pixels1 - pixels2        
        euclid = np.array( [np.sqrt(d**2) for d in diff] ).sum()
        #print(euclid)
        euclidNorm = round(euclid/(len(diff)*256),10)
        
        #print('Returning pixel difference...')
        return euclidNorm
    
    def compareXformsBasic(self, image1, image2):
        '''Applies all of the 'basic' transformations to image1 and 
        compares the results to image2. 
        
        Return all of the transformations that result in image2.
        '''
        
        # Contains the diffLogic results for each of the transformations
        # {'transformation': (transformation ID, diffLogic result)}
        xforms = dict()
        
        # Flips image about the vertical central axis, "left to right";
        # transformation ID = 01 for FLIP_LEFT_RIGHT
        image1x01 = image1.transpose(method=Image.FLIP_LEFT_RIGHT)
        t1x01 = self.diffLogicBasic(image1x01, image2)
        xforms.update({'FLIP_LEFT_RIGHT': (1, t1x01)})
        
        # Flips image about the horizontal axis
        # transformation ID = 02 for FLIP_TOP_BOTTOM
        image1x02 = image1.transpose(method=Image.FLIP_TOP_BOTTOM)
        t1x02 = self.diffLogicBasic(image1x02, image2)
        xforms.update({'FLIP_TOP_BOTTOM': (2, t1x02)})
        
        # Rotates images counterclockwise 90 degrees
        # transformation ID = 03 for ROTATE_90
        image1x03 = image1.transpose(method=Image.ROTATE_90)
        t1x03 = self.diffLogicBasic(image1x03, image2)
        xforms.update({'ROTATE_90': (3, t1x03)})
        
        # Rotates images counterclockwise 180 degrees
        # transformation ID = 04 for ROTATE_180
        image1x04 = image1.transpose(method=Image.ROTATE_180)
        t1x04 = self.diffLogicBasic(image1x04, image2)
        xforms.update({'ROTATE_180': (4, t1x04)})
        
        # Rotates images counterclockwise 270 degrees
        # transformation ID = 05 for ROTATE_270
        image1x05 = image1.transpose(method=Image.ROTATE_270)
        t1x05 = self.diffLogicBasic(image1x05, image2)
        xforms.update({'ROTATE_270': (5, t1x05)})
        
        # Reflects image about the y = -x 'transpose' axis
        # transformation ID = 06 for TRANSVERSE
        image1x06 = image1.transpose(method=Image.TRANSPOSE)
        t1x06 = self.diffLogicBasic(image1x06, image2)
        xforms.update({'TRANSPOSE': (6, t1x06)})
        
        # method=Image.TRANSVERSE raises exception when run against 
        # bonnie environment
        #
        # Reflects image about the y = x 'transverse' axis
        # transformation ID = 07 for TRANSVERSE
        #image1x07 = image1.transpose(method=Image.TRANSVERSE)
        #t1x07 = self.diffLogicBasic(image1x07, image2)
        #xforms.update({'TRANSVERSE': (7, t1x07)})
        
        # Identify which transformations are valid and then rank according 
        # to heirarchy. Returns the list of valid transformations order in 
        # accordance with the heirarchy. The heirarchy is represented by 
        # the order in which the transformations are checked above; e.g., 
        # if FLIP_LEFT_RIGHT is valid, it will always be the first element
        # in the list xformsMatch.
        xformsMatch = list()                
        for xf in xforms:
            if xforms[xf][1] == 'exact':
                xformsMatch.append(xf)
        
        return xformsMatch
        
    def applyXformBasic(self, image, xform):
        '''Applies the specified basic transformation on the image and returns
        the resulting image.
        '''
        
        # If I could dynamically assign the parameter, method=Image.xxx, 
        # to the .transpose method, this method would not be necessary.
        if xform == 'FLIP_LEFT_RIGHT':
            result = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        elif xform == 'FLIP_TOP_BOTTOM':
            result = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        elif xform == 'ROTATE_90':
            result = image.transpose(method=Image.ROTATE_90)
        elif xform == 'ROTATE_180':
            result = image.transpose(method=Image.ROTATE_180)
        elif xform == 'ROTATE_270':
            result = image.transpose(method=Image.ROTATE_270)
        elif xform == 'TRANSPOSE':
            result = image.transpose(method=Image.TRANSPOSE)
        elif xform == 'TRANSVERSE':
            result = image.transpose(method=Image.TRANSVERSE)
        
        return result
    
    def diffLogicBasic(self, image1, image2):
        
        pdiff12 = self.pixelDifference(image1, image2)
        #print('Pixel difference =', pdiff12)
        hdiff12 = self.histDifference(image1, image2)
        #print('Histogram difference =', hdiff12)
        
        # Basis for difference tolerance:
        # Basic Problem B-04 2x2 Basic Problems B True True
        # Pixel difference between A and B = 0.1894982379
        # Pixel difference between A and C = 0.1892480976
        # Histogram difference between A and B = 0.001159668
        # Histogram difference between A and C = 0.0014648438
        # Therefore, set difference tolerance to 0.01
        #
        # Reset tolerance to 0.03: basis is Basic Problem B-06 A-C transformation
        #
        # Logic flow for what histDifference and pixelDifference mean
        if pdiff12 < 0.03:
            # The two images are near-exact duplicates
            relationship = 'exact'
        else:
            if hdiff12 < 0.01:
                # The two images are not near-exact duplicates, but
                # have the near-same shape. Therefore, the two images are 
                # likely rotations or transpositions of one another
                relationship = 'shape'
            else:
                relationship = 'neither'
        
        return relationship
    
    def findAnswer(self, problem, image, answers):
        '''Compares the candidate answer to the images in the set of answers.
        '''
        answerFiles = {a:problem.figures[a].visualFilename for a in answers}
        
        # Check to ensure only one matching answer is returned
        count = 0
        answer = list()
        for af in answerFiles:
            imageAnswer = Image.open(answerFiles[af])
            imageAnswer = imageAnswer.convert('L')
            rel = self.diffLogicBasic(image, imageAnswer)
            if rel == 'exact':
                count += 1
                answer.append(af)
        
        # Only one answer found
        if count == 1:
            return answer[0]
        # No answers found
        elif count == 0:
            return False
        else:
            #print('Multiple answers found. Consider reducing tolerance.')
            # Return the first answer found
            return answer[0]