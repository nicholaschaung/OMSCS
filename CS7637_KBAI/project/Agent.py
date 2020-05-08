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
from PIL import ImageFilter
from PIL import ImageChops
import numpy as np
import timeit
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
        t0 = timeit.default_timer()
        
        # This loop writes the image data for each image to .txt files
        """ for f in problem.figures:
            image = Image.open(problem.figures[f].visualFilename)
            image = image.convert('L')
            filename = problem.figures[f].name + '_bw.txt'
            self.writeData(image, filename) """
        
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
            answer = self.twoByTwo(layout, problem)
            return answer
        
        else:
            answer = self.threeByThree(layout, problem)
            t1 = timeit.default_timer()
            print('CPU time elapsed to calculate: %s' % str(t1-t0))
            return answer        
        #return -1

    def twoByTwo(self, layout, problem):
        '''Solves 2x2 problems
        '''

        # Initial (rudimentary) problem solving methods developed 
        # for 2x2 methods removed.
        # If this agent is to be used to re-visit 2x2 methods, the 
        # original 2x2 methods should be re-worked with the framework 
        # of the 3x3 methods.

        return -1
    
    def threeByThree(self, layout, problem):
        '''Solves 3x3 problems
        '''
        question = layout[problem.problemType][0]
        answers = layout[problem.problemType][1]
        
        # The dict images will collect the images extracted from the problem.
        # The index values are from the tuple 'question' and the image objects 
        # from the problem.figures object.
        images = dict()
        # Extra pixel information in original RGBA format not necessary 
        # and converting to BW 8-pixel reduces operations
        for q in question:
            image = Image.open(problem.figures[q].visualFilename)
            image = image.convert('L')
            images.update({q: image})
        
        # Default answer is False.
        # Problem must be explicitly solved to return otherwise.
        answerDefault = -1
        
        # Perform check to determine if there is diagonal similarity.
        # Check for A-E-I diagonal similarity first
        print('Attempting to solve by diagonal similarity...')
        # tolerance = 0.02 based on Basic Problem D-11
        # hfAE = 0.9936474936, hfBF = 0.983743543
        AEI = self.checkDiagonal(images, 'AEI', 0.02)
        if AEI:
            print('A-E-I diagonal similarity found.')
            # tolerance = 0.03 based on Basic Problem D-02
            # expected hf = 1.0, answer hf = 0.9866844208
            answerAEI, diffAEI = self.findAnswerHist(problem, answers, images['E'], 1.0, 0.03)
            if answerAEI:
                return int(answerAEI)
        else:
            CEG = self.checkDiagonal(images, 'CEG', 0.02)
            if CEG:
                print('C-E-G diagonal similarity found.')
                answerCEG, diffCEG = self.findAnswerHist(problem, answers, images['D'], 1.0, 0.03)
                if answerCEG:
                    return int(answerCEG)
        
        # Perform check to determine if there is row-element addition.
        print('Attempting to solve by row-element addition...')
        # tolerance = 0.06 based on Basic Problem E-03
        # hfABC = 0.9484649123, hfBF = 
        addABC = self.checkAddition(images, 'ABC', 0.06)
        if addABC:
            print('A + B = C relationship found.')
            addGHI = ImageChops.darker(images['G'], images['H'])
            #addGHI.show()
            # tolerance = 0.03 based on Basic Problem E-03
            # expected hf = 1.0, answer hf = 0.9748848891
            answerAddGHI, diffAddGHI = self.findAnswerHist(problem, answers, addGHI, 1.0, 0.03)
            if answerAddGHI:
                return int(answerAddGHI)
        
        # Perform check to determine if there is row-element XOR operation.
        print('Attempting to solve by row-element XOR...')
        # tolerance = 0.25 based on Basic Problem E-08
        # A XOR B = C pixel difference: 0.241424956
        xorABC = self.checkXOR(images, 'ABC', 0.25)
        if xorABC:
            print('A XOR B = C relationship found.')
            xorGHI = ImageChops.difference(images['G'], images['H'])
            xorGHI = ImageChops.invert(xorGHI)
            #xorGHI.show()
            # tolerance = 0.22 based on Basic Problem E-08
            # Difference between 1 0.2170852073
            answerXorGHI, diffXorGHI = self.findAnswer(problem, xorGHI, answers, 0.22)
            if answerXorGHI:
                return int(answerXorGHI)
        
        # Perform check to see if there is row-element similarity and/or 
        # column-element similarity.
        print('Attempting to solve by row and column similarity...')
        # tolerance = 0.16 based on Basic Problem D-01
        # AB-BC pixel difference: 0.1573986462
        rowSimABC = self.checkRowSim(images, 0.16)
        if rowSimABC:
            print('Row-element similarity found.')
            # tolerance = 0.11 based on Basic Problem D-04
            # AD-DG pixel difference: 0.1018414867
            colSimADG = self.checkColSim(images, 0.11)
            if colSimADG:
                print('Column-element similarity found.')
                imageGH = ImageChops.lighter(images['G'], images['H'])
                #imageGH.show()
                imageCF = ImageChops.lighter(images['C'], images['F'])
                #imageCF.show()
                imageGHCF = ImageChops.darker(imageGH, imageCF)
                #imageGHCF.show()
                # tolerance = 0.17 based on Basic Problem D-04
                # Difference between 1 = 0.1626825653
                answerSimGHCF, diffSimGHCF = self.findAnswer(problem, imageGHCF, answers, 0.17)
                if answerSimGHCF:
                    return int(answerSimGHCF)
                #
                # ONLY ROW SIMILARITY FOUND FOR D-06
                # INCLUDE DIAGONAL SIMILARITY TO PROPERLY SOLVE PROBLEM
                #
            else:
                # tolerance = 0.11 based on Basic Problem D-06
                # DH-HC pixel difference: 0.0996550969
                diagSimDHC = self.checkDiagSim(images, 'DHC', 0.18)
                if diagSimDHC:
                    print('D-H-C diagonal similarity found.')
                    imageGH = ImageChops.lighter(images['G'], images['H'])
                    #imageGH.show()
                    imageAE = ImageChops.lighter(images['A'], images['E'])
                    #imageAE.show()
                    imageGHAE = ImageChops.darker(imageGH, imageAE)
                    #imageGHAE.show()
                    # tolerance = 0.10 based on Basic Problem 
                    # 
                    answerSimGHAE, diffSimGHAE = self.findAnswer(problem, imageGHAE, answers, 0.10)
                    if answerSimGHAE:
                        return int(answerSimGHAE)
                else:
                    print('Only row-element similarity found.')
                    # tolerance = 0.10 based on Basic Problem D-06
                    # Difference between 5 0.1094138842
                    answerSimABC, diffSimABC = self.findAnswer(problem, images['H'], answers, 0.10)
                    if answerSimABC:
                        return int(answerSimABC)
        # Check for column similarity. No basic problems involved column 
        # exhibit only column similarity, but test problems may involve such.
        else:
            colSimADG = self.checkColSim(images, 0.16)
            if colSimADG:
                # No need to also check for row similarity, since column 
                # similarity is checked for after row similarity is found.
                print('Only column-element similarity found.')
                answerSimADG, diffSimADG = self.findAnswer(problem, images['F'], answers, 0.16)
                if answerSimADG:
                    return int(answerSimADG)
        
        # Perform check to see if there is compound diagonal similarity.
        print('Attempting to solve by compound diagonal similarity...')
        # tolerance = 0.16 based on Basic Problem D-07
        # CE-EG pixel difference: 0.1316159417
        diagSimCEG = self.checkDiagSim(images, 'CEG', 0.18)
        if diagSimCEG:
            print('C-E-G diagonal similarity found.')
            # tolerance = 0.23 based on Basic Problem D-07
            # BF-FG pixel difference: 0.2210633189
            diagSimBFG = self.checkDiagSim(images, 'BFG', 0.23)
            if diagSimBFG:
                print('B-F-G diagonal similarity found.')
                imageAE = ImageChops.lighter(images['A'], images['E'])
                #imageAE.show()
                imageBD = ImageChops.lighter(images['B'], images['D'])
                #imageBD.show()
                imageAEBD = ImageChops.darker(imageAE, imageBD)
                #imageAEBD.show()
                # tolerance = 0.16 based on Basic Problem D-07
                # Difference between 1 0.1591640588
                answerDiagAEBD, diffDiagAEBD = self.findAnswer(problem, imageAEBD, answers, 0.16)
                if answerDiagAEBD:
                    return int(answerDiagAEBD)
        
        return answerDefault
    
    def writeData(self, image, filename):
        '''Writes the data from np.array() for the specified image filename
        to a .txt file.
        '''
        with open(filename, 'w') as imageData:
            data = np.array(image)
            # data.shape = (184, 184) for all images,
            # where each element is a numpy.uint8 between [0,255].
            for d in data:
                imageData.write(str(d) + '\n')

    def histFactor(self, image1, image2):
        '''Calculates the ratio between two image histogram vectors. 
        
        Each histogram vector is sequence of the counts of pixels in an image.
        '''        
        
        h1 = np.array(image1.histogram())
        h2 = np.array(image2.histogram())
        #with open('histdata.txt', 'w') as histData:
            #histData.write(str(h1))
        
        # Factor is a measure of how many more "filled in" pixels, 
        # pixel value = 0, image2 has compared to image1. If image2 has more 
        # black pixels, pixel value = 0, factor > 1.0.
        # (image1 black pixels) * factor =  image2 black pixels
        factor = round(h2[0]/h1[0], 10)
        
        return factor
    
    def pixelDifference(self, image1, image2):
        '''Calculates the Euclidean distance between two image vectors. 
        
        Each image vector is a complete sequence of pixel data.
        '''
        
        # Not converting the images to type='L' and computing the generalized 
        # Euclidean distance, Minkowski distance, results in a computation time 
        # for one distance calculation exceeding the total problem solve time 
        # when using converted type='L' images. Therefore, continue converting 
        # images to type='L'.
        #p1 = np.array(image1)
        #p2 = np.array(image2)
        #di = cdist(pixels1, pixels2, 'minkowski', 3)**2
        
        # Applies a Gaussian blur to mitigate effects of slight image offsets 
        # resulting in non-meaningful image differences resulting in larger
        # Euclidean distance values than justified.
        image1 = image1.filter(ImageFilter.GaussianBlur(radius=1))
        #image1.show()
        image2 = image2.filter(ImageFilter.GaussianBlur(radius=1))
        #image2.show()
        
        pixels1 = np.array(image1)
        pixels2 = np.array(image2)
        
        #print('Calculating pixel difference...')
        diff = np.subtract(pixels1, pixels2)
        #image12 = Image.fromarray(diff)
        #image12.show()
        # Current structure of diff is an instance of <class 'numpy.ndarray'>.
        # This structure has shown to limit the square calculation to values 
        # less than 255. Therefore, convert the 2-dimensional ndarray to a 
        # 1-dimension list of type list, where each value can be properly 
        # squared to values greater than 255 if applicable.
        diffList = list()
        for d in diff:
            for dd in d:
                diffList.append(dd)
        # diffList is now a list of length 33856 = 184*184.
        euclid = np.sqrt( np.array([d**2 for d in diffList]).sum() )
        euclidNorm = round( euclid/(np.sqrt(len(diffList))*255), 10 )
        #print(euclidNorm)
        
        #print('Returning pixel difference...')
        return euclidNorm
    
    def findAnswer(self, problem, image, answers, tolerance):
        '''Compares the candidate answer to the images in the set of answers.
        '''
        answerFiles = {a:problem.figures[a].visualFilename for a in answers}
        
        matches = list()
        diffs = list()
        for af in answerFiles:
            imageAnswer = Image.open(answerFiles[af])
            imageAnswer = imageAnswer.convert('L')
            diff = self.pixelDifference(image, imageAnswer)
            print('Difference between', af, diff)
            if diff < tolerance:
                matches.append(af)
                diffs.append(diff)
        
        # Only one answer found
        if len(matches) == 1:
            return matches[0], diffs[0]
        # No answers found
        elif len(matches) == 0:
            return False, False
        else:
            print('Multiple answers found. Consider reducing tolerance.')
            
            # Zips matches and diffs together in order to order the results 
            # by difference; least to greatest
            diffsMatches = zip(diffs, matches)
            diffsMatches = sorted(diffsMatches, key=lambda x: x[0])
            diffs = [d for d, m in diffsMatches]
            matches = [m for d, m in diffsMatches]
            
            # Return the first answer in matches and its difference value; 
            # i.e., the answer with the lowest difference
            return matches[0], diffs[0]
    
    def findAnswerHist(self, problem, answers, image, histFactor, tolerance):
        '''Finds the answer with the most similar histogram difference to the 
        passed histFactor.
        '''
        print('Finding answer based on histogram factor...')
        answerFiles = {a:problem.figures[a].visualFilename for a in answers}
        
        matches = list()
        diffs = list()
        for af in answerFiles:
            imageAnswer = Image.open(answerFiles[af])
            imageAnswer = imageAnswer.convert('L')
            hfImageAnswer = self.histFactor(image, imageAnswer)
            print('Histogram factor between image and answer', af, hfImageAnswer)
            diff = np.abs(histFactor - hfImageAnswer)
            if diff < tolerance:
                matches.append(af)
                diffs.append(diff)
        
        if len(matches) == 0:
            return False, False
        else:
            # Zips matches and diffs together in order to order the results 
            # by difference; least to greatest
            diffsMatches = zip(diffs, matches)
            diffsMatches = sorted(diffsMatches, key=lambda x: x[0])
            diffs = [d for d, m in diffsMatches]
            matches = [m for d, m in diffsMatches]
            
            # Return the first answer in matches and its difference value; 
            # i.e., the answer with the lowest difference
            return matches[0], diffs[0]
    
    def checkDiagonal(self, images, diagType, tolerance):
        '''Checks problem to determine if diagonal similarity exists.
        
        Types of similarity can be: A-E-I, C-E-G
        '''
        diagonal = False
        
        if diagType == 'AEI':
            hfAE = self.histFactor(images['A'], images['E'])
            print('A-E histogram factor:', hfAE)
            # Normalize hf to difference from 1.0; "histogram factor difference"
            hfdAE = np.abs(1.0 - hfAE)
            if hfdAE < tolerance:
                # Verify diagonal similarity against B-F-G diagonal.
                hfBF = self.histFactor(images['B'], images['F'])
                print('B-F histogram factor:', hfBF)
                hfdBF = np.abs(1.0 - hfBF)
                if hfdBF < tolerance:
                    diagonal = True
        elif diagType == 'CEG':
            hfCE = self.histFactor(images['C'], images['E'])
            print('C-E histogram factor:', hfCE)
            # Normalize hf to difference from 1.0; "histogram factor difference"
            hfdCE = np.abs(1.0 - hfCE)
            if hfdCE < tolerance:
                # Verify diagonal similarity against F-H-A diagonal.
                hfFH = self.histFactor(images['F'], images['H'])
                print('F-H histogram factor:', hfFH)
                hfdFH = np.abs(1.0 - hfFH)
                if hfdFH < tolerance:
                    diagonal = True
        
        return diagonal
    
    def checkRowSim(self, images, tolerance):
        '''Checks problem to determine if row similarity exists.
        '''
        row = False
        
        imageAB = ImageChops.lighter(images['A'], images['B'])
        #imageAB.show()
        imageBC = ImageChops.lighter(images['B'], images['C'])
        #imageBC.show()
        
        pdABBC = self.pixelDifference(imageAB, imageBC)
        print('AB-BC pixel difference:', pdABBC)
        
        # If the average value of the common image is below threshold, 
        # it may be indicating the images have nothing in common.
        # If both AD and DG have nothing in common, their difference could 
        # trigger a "false positive" in continuing to solve the problem.
        pixelsAB = np.array(imageAB)
        pixelsBC = np.array(imageBC)
        
        # Try blurring or shifting images and re-calculating the difference.
        if pdABBC > tolerance:
            imageAblur = images['A'].filter(ImageFilter.GaussianBlur(radius=1))
            imageBblur = images['B'].filter(ImageFilter.GaussianBlur(radius=1))
            imageCblur = images['C'].filter(ImageFilter.GaussianBlur(radius=1))
            imageABblur = ImageChops.lighter(imageAblur, imageBblur)
            #imageABblur.show()
            imageBCblur = ImageChops.lighter(imageBblur, imageCblur)
            #imageBCblur.show()
            
            pdABBCblur = self.pixelDifference(imageABblur, imageBCblur)
            print('AB-BC blurred pixel difference:', pdABBCblur)
            #imageAx, imageB = self.shiftImage(images['A'], images['B'], tolerance)
            #if imageAx and imageB:
                #row = True
            
            if pdABBCblur < tolerance:
                row = True
        # See above comment when np.array objects were created.
        # tolerance = 250 based on Basic Problem D-01 (see checkColSim)
        elif pdABBC < tolerance and pixelsAB.mean() < 250 and pixelsBC.mean() < 250:
            row = True
        
        return row

    def checkColSim(self, images, tolerance):
        '''Checks problem to determine if column similarity exists.
        '''
        column = False
        
        imageAD = ImageChops.lighter(images['A'], images['D'])
        #imageAD.show()
        imageDG = ImageChops.lighter(images['D'], images['G'])
        #imageDG.show()
        
        pdADDG = self.pixelDifference(imageAD, imageDG)
        print('AD-DG pixel difference:', pdADDG)
        
        # If the average value of the common image is below threshold, 
        # it may be indicating the images have nothing in common.
        # If both AD and DG have nothing in common, their difference could 
        # trigger a "false positive" in continuing to solve the problem.
        pixelsAD = np.array(imageAD)
        pixelsDG = np.array(imageDG)
        #print('Mean value of common A-D image:', pixelsAD.mean())
        #print('Mean value of common D-G image:', pixelsDG.mean())
        
        # Try blurring or shifting images and re-calculating the difference.
        if pdADDG > tolerance:
            imageAblur = images['A'].filter(ImageFilter.GaussianBlur(radius=1))
            imageDblur = images['D'].filter(ImageFilter.GaussianBlur(radius=1))
            imageGblur = images['G'].filter(ImageFilter.GaussianBlur(radius=1))
            imageADblur = ImageChops.lighter(imageAblur, imageDblur)
            #imageADblur.show()
            imageDGblur = ImageChops.lighter(imageDblur, imageGblur)
            #imageDGblur.show()
            
            pdADDGblur = self.pixelDifference(imageADblur, imageDGblur)
            print('AD-DG blurred pixel difference:', pdADDGblur)
            #imageAx, imageB = self.shiftImage(images['A'], images['B'], tolerance)
            #if imageAx and imageB:
                #row = True
            
            if pdADDGblur < tolerance:
                column = True
        # See above comment when np.array objects were created.
        # tolerance = 250 based on Basic Problem D-01
        # Mean value of common A-D image: 253.83007443289225
        # Mean value of common D-G image: 255.0
        elif pdADDG < tolerance and pixelsAD.mean() < 250 and pixelsDG.mean() < 250:
            column = True
        
        return column

    def checkDiagSim(self, images, diagType, tolerance):
        '''Checks problem to determine if diagonal similarity exists.
        
        This method looks for similarity within the image and not just at 
        the image as a whole, as checkDiagonal does.
        '''
        diagonal = False
        
        if diagType == 'CEG':
            imageCE = ImageChops.lighter(images['C'], images['E'])
            #imageCE.show()
            imageEG = ImageChops.lighter(images['E'], images['G'])
            #imageEG.show()
            
            pdCEEG = self.pixelDifference(imageCE, imageEG)
            print('CE-EG pixel difference:', pdCEEG)
            
            # If the average value of the common image is below threshold, 
            # it may be indicating the images have nothing in common.
            pixelsCE = np.array(imageCE)
            pixelsEG = np.array(imageEG)
            #print('Mean value of common A-D image:', pixelsAD.mean())
            #print('Mean value of common D-G image:', pixelsDG.mean())
            
            # See above comment when np.array objects were created.
            # tolerance = 250 based on Basic Problem 
            # 
            if pdCEEG < tolerance and pixelsCE.mean() < 250 and pixelsEG.mean() < 250:
                diagonal = True
        
        elif diagType == 'BFG':
            imageBF = ImageChops.lighter(images['B'], images['F'])
            #imageBF.show()
            imageFG = ImageChops.lighter(images['F'], images['G'])
            #imageFG.show()
            
            pdBFFG = self.pixelDifference(imageBF, imageFG)
            print('BF-FG pixel difference:', pdBFFG)
            
            # If the average value of the common image is below threshold, 
            # it may be indicating the images have nothing in common.
            pixelsBF = np.array(imageBF)
            pixelsFG = np.array(imageFG)
            #print('Mean value of common A-D image:', pixelsAD.mean())
            #print('Mean value of common D-G image:', pixelsDG.mean())
            
            # See above comment when np.array objects were created.
            # tolerance = 250 based on Basic Problem 
            # 
            if pdBFFG < tolerance and pixelsBF.mean() < 250 and pixelsFG.mean() < 250:
                diagonal = True

        elif diagType == 'DHC':
            imageDH = ImageChops.lighter(images['D'], images['H'])
            #imageDH.show()
            imageHC = ImageChops.lighter(images['H'], images['C'])
            #imageHC.show()
            
            pdDHHC = self.pixelDifference(imageDH, imageHC)
            print('DH-HC pixel difference:', pdDHHC)
            
            # If the average value of the common image is below threshold, 
            # it may be indicating the images have nothing in common.
            pixelsDH = np.array(imageDH)
            pixelsHC = np.array(imageHC)
            #print('Mean value of common A-D image:', pixelsAD.mean())
            #print('Mean value of common D-G image:', pixelsDG.mean())
            
            # See above comment when np.array objects were created.
            # tolerance = 250 based on Basic Problem 
            # 
            if pdDHHC < tolerance and pixelsDH.mean() < 250 and pixelsHC.mean() < 250:
                diagonal = True
        
        return diagonal

    def shiftImage(self, image1, image2, tolerance):
        '''Shifts image1 in each direction and checks if the histogram factor
        difference is within tolerance.
        
        If a shifted image is within tolerance, return the shifted image and 
        the other image.
        '''
        xshifts = (-2, 2)
        #yshifts = (-2, 2)
        for xs in xshifts:
            image1x = ImageChops.offset(image1, xoffset=xs, yoffset=None)
            image1x2 = ImageChops.lighter(image1x, image2)
            self.histFactor(image1x2, image2)        
    
    def checkAddition(self, images, addSequence, tolerance):
        '''Checks problem to determine if image addition relationship exists.
        
        addSequence specifies the addition sequence; e.g., 'ABC' implies 
        A + B = C, 'ACB' implies A + C = B.
        '''
        addition = False
        
        if addSequence == 'ABC':
            imageAB = ImageChops.darker(images['A'], images['B'])
            hfABC = self.histFactor(imageAB, images['C'])
            print('A + B = C histogram factor:', hfABC)
            # Normalize hf to difference from 1.0; "histogram factor difference"
            hfdABC = np.abs(1.0 - hfABC)
            if hfdABC < tolerance:
                # Verify D + E = F
                imageDE = ImageChops.darker(images['D'], images['E'])
                hfDEF = self.histFactor(imageDE, images['F'])
                print('D + E = F histogram factor:', hfDEF)
                hfdDEF = np.abs(1.0 - hfDEF)
                if hfdDEF < tolerance:
                    addition = True
        elif addSequence == 'ACB':
            imageAC = ImageChops.darker(images['A'], images['C'])
            hfACB = self.histFactor(imageAC, images['B'])
            print('A + C = B histogram factor:', hfACB)
            # Normalize hf to difference from 1.0; "histogram factor difference"
            hfdACB = np.abs(1.0 - hfACB)
            if hfdACB < tolerance:
                # Verify D + F = E
                imageDF = ImageChops.darker(images['D'], images['F'])
                hfDFE = self.histFactor(imageDF, images['E'])
                print('D + F = E histogram factor:', hfDFE)
                hfdDFE = np.abs(1.0 - hfDFE)
                if hfdDFE < tolerance:
                    addition = True
        
        return addition
    
    def checkXOR(self, images, xorSequence, tolerance):
        '''Checks problem to determine if image XOR relationship exists.
        
        xorSequence specifies the XOR sequence; e.g., 'ABC' implies 
        A XOR B = C, 'ACB' implies A XOR C = B.
        '''
        XOR = False
        
        if xorSequence == 'ABC':
            imageAB = ImageChops.difference(images['A'], images['B'])
            imageAB = ImageChops.invert(imageAB)
            #imageAB.show()
            pdABC = self.pixelDifference(imageAB, images['C'])
            print('A XOR B = C pixel difference:', pdABC)
            
            # If the average value of the common image is below threshold, 
            # it may be indicating the images have nothing in common.
            pixelsAB = np.array(imageAB)
            #print('Mean value of A XOR B image:', pixelsAB.mean())
            
            if pdABC < tolerance and pixelsAB.mean() < 247:
                XOR = True
        
        return XOR
