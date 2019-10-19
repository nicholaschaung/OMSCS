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
#from scipy.spatial.distance import cdist

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
        
        # This loop writes the image data for each image to .txt files
#        for f in problem.figures:
#            image = Image.open(problem.figures[f].visualFilename)
#            image = image.convert('L')
#            filename = problem.figures[f].name + '.txt'
#            self.writeData(image, filename)
        
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
            
            #imageA.show()
            #xform = (1, 0, 0, 0, 1, 0)
            #imageAx = self.applyXformAffine(imageA, xform)
            #imageAx.show()
                        
            # Test all the 'basic' transformations and return the transformations
            # that map A-B and A-C within the specified tolerance.
            #
            # For Problem 6, within self.compareXformsBasic(imageA, imageC):
            # [0.0027637734, 0.022322945, 0.0279591782, 0.0281285537, 0.0481293936, 0.0656495987]
            # ['FLIP_TOP_BOTTOM', 'ROTATE_90', 'ROTATE_270', 'FLIP_LEFT_RIGHT', 'ROTATE_180', 'TRANSPOSE']
            # For Problem 11, within self.compareXformsBasic(imageA, imageB):
            # [0.0254438144, 0.0971996978, 0.0975686778, 0.0975686778, 0.1165908785, 0.1165982627]
            # ['FLIP_LEFT_RIGHT', 'TRANSPOSE', 'ROTATE_90', 'ROTATE_270', 'FLIP_TOP_BOTTOM', 'ROTATE_180']
            # Therefore, the tolerance would have to be set > 0.0279591782 
            # to return the correct answer for Problem 6, 'ROTATE_270', but 
            # would have to be set < 0.0254438144 to eliminate the 
            # incorrect answer for Problem 11, 'FLIP_LEFT_RIGHT'.
            xformsBasicAB, xformsBasicABnames = self.compareXformsBasic(imageA, imageB, 0.03, 0.022)
            #print('Indicative transformation(s) for A-B is/are:', xformsBasicAB)
            xformsBasicAC, xformsBasicACnames = self.compareXformsBasic(imageA, imageC, 0.03, 0.022)
            #print('Indicative transformation(s) for A-C is/are:', xformsBasicACnames)
            
            # If they exists, apply the indicative transformations on C and B.
            # Then check if the resulting image is one of the answers.
            # If the result is not one of the answers, iterate through the list
            # of valid basic transformations.
            if xformsBasicAB:
                count = 0
                answerFromAB = False
                while count < len(xformsBasicAB) and (not answerFromAB):
                    imageDfromAB = self.applyXformBasic(imageC, xformsBasicABnames[count])
                    answerFromAB = self.findAnswer(problem, imageDfromAB, answers, 0.03)
                    #print('Answer from A-B transformation:', answerFromAB)
                    count += 1
            #else:
                #print('No basic indicative transformation found for A-B.')
            
            if xformsBasicAC:
                count = 0
                answerFromAC = False
                while count < len(xformsBasicAC) and not answerFromAC:
                    imageDfromAC = self.applyXformBasic(imageB, xformsBasicACnames[count])
                    answerFromAC = self.findAnswer(problem, imageDfromAC, answers, 0.03)
                    #print('Answer from A-C transformation:', answerFromAC)
                    count += 1
            #else:
                #print('No basic indicative transformation found for A-C.')
            
            # Both transformations were identified
            #
            # solvedByBasic tracks whether or not the basic methods have solved
            # the problem. If they haven't, the sequence continues to solving
            # by pixel.
            solvedByBasic = False
            if xformsBasicAB and xformsBasicAC:
                # If both transformations resulted in the same answer, 
                # that answer is the unambiguous solution.
                if answerFromAB == answerFromAC:
                    solvedByBasic = True
                    return int(answerFromAB)
                # If both transformations were identified, but one did not 
                # result in an answer, choose the one that did find an answer.
                elif answerFromAB and not answerFromAC:
                    solvedByBasic = True
                    return int(answerFromAB)
                elif answerFromAC and not answerFromAB:
                    solvedByBasic = True
                    return int(answerFromAC)
                # If both transformations found answers but they differed, 
                # choose the A-B transformation (arbitrary heirarchy based 
                # upon what I perceive to be the more apparent relationship).
                elif answerFromAB != answerFromAC:
                    solvedByBasic = True
                    return int(answerFromAB)
                # Neither transformation resulted in an available answer
                #else:
                    #print('No matching answer image found. No final answer from the basic solver.')
                    #return -1
            # Only the A-B transformation was identified
            elif xformsBasicAB:
                if answerFromAB:
                    #print('Only A-B transformation exists and it produces an answer.')
                    solvedByBasic = True
                    return int(answerFromAB)
                #else:
                    #print('No valid transformation. No final answer from the basic solver.')
                    #return -1
            # Only the A-C transformation was identified
            elif xformsBasicAC:
                if answerFromAC:
                    #print('Only A-C transformation exists and it produces an answer.')
                    solvedByBasic = True
                    return int(answerFromAC)
                #else:
                    #print('No valid transformation. No final answer from the basic solver.')
                    #return -1
            #else:
                #print('No valid transformation. No final answer from the basic solver.')
                #return -1
        
            # If the previous basic transformation solve methods did not work,
            # attempt pixel-wise transformation comparison
            #
            # solvedByPixel tracks whether or not the pixel method solved it.
            solvedByPixel = False
            if solvedByBasic == False:
                print('Attempting to solve by pixel difference..')
                xformPixelAB = self.generateXformPixel(imageA, imageB)
                xformPixelAC = self.generateXformPixel(imageA, imageC)
                
                #with open('difference.txt', 'w') as imageData:
                    #for x in xformPixelAB:
                        #imageData.write(str(x) + '\n')
                
                imageDfromABpixel = self.applyXformPixel(imageC, xformPixelAB)
                #imageDfromABpixel.show()
                imageDfromACpixel = self.applyXformPixel(imageB, xformPixelAC)
                #imageDfromACpixel.show()
                
                answerFromABpixel = self.findAnswer(problem, imageDfromABpixel, answers, 0.05)
                #print(answerFromABpixel)
                answerFromACpixel = self.findAnswer(problem, imageDfromACpixel, answers, 0.05)
                #print(answerFromACpixel)
                
                # If both answers equal each other and are valid
                if answerFromABpixel and (answerFromABpixel == answerFromACpixel):
                    solvedByPixel = True
                    return int(answerFromABpixel)
                #else:
                    #print('Pixel difference method did not find answer.')
            
            if solvedByBasic == False and solvedByPixel == False:
                #print('Neither method found an answer.')
                return -1
        
        # Anything not 2x2 not handled at this point
        else:
            return -1
        
        #return -1

    def writeData(self, image, filename):
        '''Writes the data from np.array() for the specified image filename
        to a .txt file.
        '''
        with open(filename, 'w') as imageData:
            data = np.array(image)
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
        
        # Not converting the images to type='L' and computing the generalized 
        # Euclidean distance, Minkowski distance, results in a computation time 
        # for one distance calculation exceeding the total problem solve time 
        # when using converted type='L' images.
        #p1 = np.array(image1)
        #print(p1.shape)
        #p2 = np.array(image2)
        #print(p2.shape)
        #di = cdist(pixels1, pixels2, 'minkowski', 3)**2
        #print(di.shape)
        
        pixels1 = np.array(list(image1.getdata()))
        pixels2 = np.array(list(image2.getdata()))
        
        #print('Calculating pixel difference...')
        diff = pixels1 - pixels2
        euclid = np.array( [np.sqrt(d**2) for d in diff] ).sum()
        euclidNorm = round( euclid/(len(diff)*256), 10 )
        
        #print('Returning pixel difference...')
        return euclidNorm
    
    def compareXformsBasic(self, image1, image2, tolerance1, tolerance2):
        '''Applies all of the 'basic' transformations to image1 and 
        compares the results to image2. 
        
        Returns all of the transformations that result in image2.
        '''
        
        # Paired lists, where the index of the name in xformNames will 
        # correspond with the index of the value in xformValues
        xformValues = list()
        xformNames = list()
        
        # Flips image about the vertical central axis, "left to right";
        # transformation ID = 01 for FLIP_LEFT_RIGHT
        image1x01 = image1.transpose(method=Image.FLIP_LEFT_RIGHT)
        diff1x01 = self.pixelDifference(image1x01, image2)
        xformValues.append(diff1x01)
        xformNames.append('FLIP_LEFT_RIGHT')
        
        # Flips image about the horizontal axis
        # transformation ID = 02 for FLIP_TOP_BOTTOM
        image1x02 = image1.transpose(method=Image.FLIP_TOP_BOTTOM)
        diff1x02 = self.pixelDifference(image1x02, image2)
        xformValues.append(diff1x02)
        xformNames.append('FLIP_TOP_BOTTOM')
        
        # Rotates images counterclockwise 90 degrees
        # transformation ID = 03 for ROTATE_90
        image1x03 = image1.transpose(method=Image.ROTATE_90)
        diff1x03 = self.pixelDifference(image1x03, image2)
        xformValues.append(diff1x03)
        xformNames.append('ROTATE_90')
        
        # Rotates images counterclockwise 180 degrees
        # transformation ID = 04 for ROTATE_180
        image1x04 = image1.transpose(method=Image.ROTATE_180)
        diff1x04 = self.pixelDifference(image1x04, image2)        
        xformValues.append(diff1x04)
        xformNames.append('ROTATE_180')
        
        # Rotates images counterclockwise 270 degrees
        # transformation ID = 05 for ROTATE_270
        image1x05 = image1.transpose(method=Image.ROTATE_270)
        diff1x05 = self.pixelDifference(image1x05, image2)
        xformValues.append(diff1x05)
        xformNames.append('ROTATE_270')
        
        # Reflects image about the y = -x 'transpose' axis
        # transformation ID = 06 for TRANSVERSE
        image1x06 = image1.transpose(method=Image.TRANSPOSE)
        diff1x06 = self.pixelDifference(image1x06, image2)
        xformValues.append(diff1x06)
        xformNames.append('TRANSPOSE')
        
        # Zips xformValues and xformNames together in order to order the 
        # results by difference; least to greatest
        valuesNames = zip(xformValues, xformNames)
        valuesNames = sorted(valuesNames, key=lambda x: x[0])
        xformValues = [v for v, n in valuesNames]
        xformNames = [n for v, n in valuesNames]
        
        # If only one value in xformValues is below the tolerance, lower the 
        # tolerance to the smaller threshold
        # The default is the tolerance1 value.
        tolerance = tolerance1
        count = 0
        for v in xformValues:
            if v <= tolerance1:
                count += 1
        if count == 1:
            tolerance = tolerance2
        
        # Pulls only the values from xformValues and xformNames < tolerance
        count = 0
        finalValues = list()
        finalNames = list()
        for v in xformValues:
            if v <= tolerance:
                finalValues.append(v)
                finalNames.append(xformNames[count])
            count += 1
        
        return finalValues, finalNames
    
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
    
    def findAnswer(self, problem, image, answers, tolerance):
        '''Compares the candidate answer to the images in the set of answers.
        '''
        answerFiles = {a:problem.figures[a].visualFilename for a in answers}
        
        # Check to ensure only one matching answer is returned
        count = 0
        matches = list()
        diffs = list()
        for af in answerFiles:
            imageAnswer = Image.open(answerFiles[af])
            imageAnswer = imageAnswer.convert('L')
            diff = self.pixelDifference(image, imageAnswer)
            if diff < tolerance:
                count += 1
                matches.append(af)
                diffs.append(diff)
        
        # Only one answer found
        if len(matches) == 1:
            return matches[0]
        # No answers found
        elif len(matches) == 0:
            return False
        else:
            print('Multiple answers found. Consider reducing tolerance.')
            
            # Zips matches and diffs together in order to order the results 
            # by difference; least to greatest
            diffsMatches = zip(diffs, matches)
            diffsMatches = sorted(diffsMatches, key=lambda x: x[0])
            diffs = [d for d, m in diffsMatches]
            matches = [m for d, m in diffsMatches]
            
            # Return the first answer in matches; i.e., the answer with the 
            # lowest difference
            return matches[0]
    
    def generateXformPixel(self, image1, image2):
        '''Generates an additive transformation that will transform image1 to 
        image 2 pixel-by-pixel.
        
        Returns an array equal in length to image1 and image2. Adding this 
        array element-by-element to np.array(image1) results in np.array(image2).
        '''
        pixels1 = np.array(image1)
        pixels2 = np.array(image2)
        xform = np.subtract(pixels2, pixels1)
        
        return xform
    
    def applyXformPixel(self, image, xform):
        '''Applies the specified transformation to the image and returns the 
        transformed image.
        
        The parameter xform should be an array of equal length to np.array(image).
        '''
        pixels = np.array(image)
        result = pixels + xform
        result = Image.fromarray(result)
        return result

    def applyXformAffine(self, image, xform):
        imageX = image.transform(image.size, Image.AFFINE, data=xform, resample=0)
        return imageX
    
