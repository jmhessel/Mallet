package cc.mallet.topics;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

import cc.mallet.classify.MaxEnt;
import cc.mallet.topics.TopicInferencer.Pair;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Dirichlet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.MatrixOps;
import gnu.trove.TIntIntIterator;

public class DMRTopicInferencer extends TopicInferencer {

	int numFeatures;
	int defaultFeatureIndex;
	MaxEnt classifier = null;
	double [] parameters;
	gnu.trove.TIntIntHashMap[] oldTypeTopicCounts = null;
	
	public DMRTopicInferencer(int[][] typeTopicCounts,  int[] tokensPerTopic, Alphabet alphabet,
			double beta, double betaSum, Instance oneInstance, MaxEnt classifier, gnu.trove.TIntIntHashMap[] oldTypeTopicCounts) {
		
		super(typeTopicCounts, tokensPerTopic, alphabet, new double[typeTopicCounts[0].length], beta, betaSum);
		System.out.println("I think that there are " + this.numTopics + " topics.");
		System.out.println("I am using a topic mask of " + this.topicMask + " " + this.topicBits);

		numFeatures = oneInstance.getTargetAlphabet().size() + 1;
		defaultFeatureIndex = this.numFeatures - 1;
		
		System.out.println("There are " + this.numFeatures  + " features.");
		
		this.classifier = classifier;
		this.parameters = classifier.getParameters();
		System.out.println(parameters.length);
		this.numTopics = typeTopicCounts[0].length;
		
		System.out.println("There are " + this.numTopics + " topics.");
		
		this.oldTypeTopicCounts = oldTypeTopicCounts;
	}
	
	
	/**
	 *  Use only the default features to set the topic prior (use no document features)
 	 */
	public void setAlphas() {

        //double[] parameters = dmrParameters.getParameters();
        
		smoothingOnlyMass = 0.0;

        // Use only the default features to set the topic prior (use no document features)
        for (int topic=0; topic < numTopics; topic++) {
            alpha[topic] = Math.exp( parameters[ (topic * numFeatures) + defaultFeatureIndex ] );
			
			smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
			cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);
        }

    }
	
	/**
	 *  Set alpha based on features in an instance
	 */
    public void setAlphas(Instance instance) {
    
        // we can't use the standard score functions from MaxEnt,
        //  since our features are currently in the Target.
        FeatureVector features = (FeatureVector) instance.getTarget();
        if (features == null) { setAlphas(); return; }
        
        //double[] parameters = dmrParameters.getParameters();
        
		smoothingOnlyMass = 0.0;
        
        for (int topic = 0; topic < numTopics; topic++) {
            alpha[topic] = parameters[topic*numFeatures + defaultFeatureIndex]
                + MatrixOps.rowDotProduct (parameters,
                                           numFeatures,
                                           topic, features,
                                           defaultFeatureIndex,
                                           null);
            
            alpha[topic] = Math.exp(alpha[topic]);

			smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
			cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);
        }
    }
	
	
	/**
	 *  Infer topics for the provided instances and
	 *   write distributions to the provided file.
	 *
	 *  @param instances
	 *  @param distributionsFile
	 *  @param numIterations The total number of iterations of sampling per document
	 *  @param thinning	  The number of iterations between saved samples
	 *  @param burnIn		The number of iterations before the first saved sample
	 *  @param threshold	 The minimum proportion of a given topic that will be written
	 *  @param max		   The total number of topics to report per document]
	 */
	public void writeInferredDistributionsAndLogLikelihoods(InstanceList instances, 
										   					File distributionsFile,
										   					File logLikelihoodFile,
										   					int numIterations, int thinning, int burnIn,
										   					double threshold, int max) throws IOException {

		PrintWriter out = new PrintWriter(distributionsFile);
		PrintWriter logLikeOut = new PrintWriter(logLikelihoodFile);
		
		out.print ("#doc name topic proportion ...\n");

		IDSorter[] sortedTopics = new IDSorter[ numTopics ];
		for (int topic = 0; topic < numTopics; topic++) {
			// Initialize the sorters with dummy values
			sortedTopics[topic] = new IDSorter(topic, topic);
		}

		if (max < 0 || max > numTopics) {
			max = numTopics;
		}

		int doc = 0;

		for (Instance instance: instances) {
			
			StringBuilder builder = new StringBuilder();
			
			//Modify this.alpha to be the correct alpha for this document.
			setAlphas(instance);
			
			Pair<double [], int []> p = getSampledDistributionAndTopics(instance, numIterations, thinning, burnIn);
			
			double [] topicDistribution = p.getFirst();
			int [] topicAssignments = p.getSecond();
			logLikeOut.print(logLikelihood(topicAssignments,instance) + "\n");
			
			builder.append(doc);
			builder.append("\t");

			if (instance.getName() != null) {
				builder.append(instance.getName()); 
			}
			else {
				builder.append("no-name");
			}

			if (threshold > 0.0) {
				for (int topic = 0; topic < numTopics; topic++) {
					sortedTopics[topic].set(topic, topicDistribution[topic]);
				}
				Arrays.sort(sortedTopics);
				
				for (int i = 0; i < max; i++) {
					if (sortedTopics[i].getWeight() < threshold) { break; }
					
					builder.append("\t" + sortedTopics[i].getID() +
								   "\t" + sortedTopics[i].getWeight());
				}
			}
			else {
				for (int topic = 0; topic < numTopics; topic++) {
					builder.append("\t" + topicDistribution[topic]);
				}
			}
			out.println(builder);
			doc++;
		}

		out.close();
	}
	
	public double logLikelihood(int[] topics, Instance instance) {
		
		double loglike = 0.0;
		FeatureSequence tokens = (FeatureSequence) instance.getData();
		int docLength = tokens.size();
		
		double alphaSum = 0.0;
		for(double a : alpha) alphaSum += a;
		
		//get the sum of each topic in this document
		int[] curDocTopicCounts = new int[this.numTopics];
		for(int i = 0; i < curDocTopicCounts.length; ++i) curDocTopicCounts[i] = 0;
		
		int topic;
		int type;
		int validLen = 0;
		// Do the P(w_i | z_i) first...
		for (int position = 0; position < docLength; position++) {
			type = tokens.getIndexAtPosition(position);
			topic = topics[position];
			// Ignore out of vocabulary terms
			if (type < numTypes && typeTopicCounts[type].length != 0) {
				validLen ++;
				curDocTopicCounts[topic] ++;
				int count = oldTypeTopicCounts[type].get(topic);
				if(count > 0) {
					loglike += Math.log(beta + count);
					loglike -= Math.log(tokensPerTopic[topic] + betaSum);
				}
			}
		}
		
		//Do the P(zi | alpha) next
		for (int position = 0; position < docLength; position++) {
			type = tokens.getIndexAtPosition(position);
			topic = topics[position];
			if (type < numTypes && typeTopicCounts[type].length != 0) {
				// Ignore out of vocabulary terms
				if (type < numTypes && typeTopicCounts[type].length != 0) {
					loglike += Math.log(curDocTopicCounts[topic] + alpha[topic]);
					loglike -= Math.log(validLen + alphaSum);
				}
			}
		}
		
		return loglike;
	}
	

	

	public static void main (String[] args) throws IOException {
		//instances, maxent model, typeTopic, tokensPerTopic

        InstanceList testing = InstanceList.load (new File(args[0]));
        
        FileInputStream fis;
        ObjectInputStream oos;
        
        MaxEnt params = null;
        
        fis = new FileInputStream(args[1]);
        oos = new ObjectInputStream(fis);
        try {
        	params = (MaxEnt)(oos.readObject());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
        
        
        int [][] typeTopicArr = null;
        fis = new FileInputStream(args[2]);
        oos = new ObjectInputStream(fis);
        try {
			typeTopicArr = (int [][]) oos.readObject();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        int [] tokensPerTopicArr = null;
        fis = new FileInputStream(args[3]);
        oos = new ObjectInputStream(fis);
        try {
        	tokensPerTopicArr = (int []) oos.readObject();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        gnu.trove.TIntIntHashMap[] oldTypeTopicCounts = null;
        fis = new FileInputStream(args[4]);
        oos = new ObjectInputStream(fis);
        try {
        	oldTypeTopicCounts = (gnu.trove.TIntIntHashMap[]) oos.readObject();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        
        double beta = .01;
        double betaSum = tokensPerTopicArr.length * beta;
        System.out.println("Using beta = " + beta + " and beta sum = " + betaSum);
        
        DMRTopicInferencer dmrti = new DMRTopicInferencer(typeTopicArr,
        		tokensPerTopicArr,
        		testing.get(0).getDataAlphabet(),
        		beta, betaSum, testing.get(0), params, oldTypeTopicCounts);
        
        File outputFile = new File("output.txt");
        File likelihoodFile = new File("likelihood.txt");
        
        dmrti.writeInferredDistributionsAndLogLikelihoods(testing, outputFile, likelihoodFile, 100, 10, 10, 0.0, -1);

	}
	
	
}
