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
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.MatrixOps;
import gnu.trove.TIntIntIterator;

public class DMRTopicInferencer extends TopicInferencer {

	int numFeatures;
	int defaultFeatureIndex;
	MaxEnt classifier = null;
	double [] parameters;
	
	public DMRTopicInferencer(int[][] typeTopicCounts, int[] tokensPerTopic, Alphabet alphabet, double[] alpha,
			double beta, double betaSum, Instance oneInstance, MaxEnt classifier) {
		super(typeTopicCounts, tokensPerTopic, alphabet, alpha, beta, betaSum);
		this.numFeatures = oneInstance.getTargetAlphabet().size() + 1;
		this.defaultFeatureIndex = numFeatures - 1;
		this.classifier = classifier;
		this.parameters = classifier.getParameters();
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
	public void writeInferredDistributions(InstanceList instances, 
										   File distributionsFile,
										   int numIterations, int thinning, int burnIn,
										   double threshold, int max) throws IOException {

		PrintWriter out = new PrintWriter(distributionsFile);
		
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
			
			
			double[] topicDistribution =
				getSampledDistribution(instance, numIterations,
									   thinning, burnIn);
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

	public static void main (String[] args) throws IOException {
		//instances, maxent model, typeTopic, tokensPerTopic

        InstanceList testing = InstanceList.load (new File(args[0]));
        
        FileInputStream fis;
        ObjectInputStream oos;
        
        MaxEnt params = null;
        gnu.trove.TIntIntHashMap[] typeTopicCounts = null;
        int[] tokensPerTopic = null;
        
        fis = new FileInputStream(args[1]);
        oos = new ObjectInputStream(fis);
        try {
        	params = (MaxEnt)(oos.readObject());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
        
        fis = new FileInputStream(args[2]);
        oos = new ObjectInputStream(fis);
        try {
        	typeTopicCounts = (gnu.trove.TIntIntHashMap[])(oos.readObject());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
        
        fis = new FileInputStream(args[3]);
        oos = new ObjectInputStream(fis);
        try {
        	tokensPerTopic = (int[])(oos.readObject());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
        
        
        System.out.println(typeTopicCounts.length);
        TIntIntIterator it = typeTopicCounts[0].iterator();
        
        System.out.println(typeTopicCounts[0].get(2));
        
        
        
        
        //DMRTopicInferencer dmrti = new DMRTopicInferencer(typeTopicCounts, tokensPerTopic,
        //		testing.get(0).getDataAlphabet(),.1, ,1, 30);
        
        //System.out.println(myMaxEnt.getParameters());

	}
	
	
}
