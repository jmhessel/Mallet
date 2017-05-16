package cc.mallet.topics.tui;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import cc.mallet.topics.TopicInferencer;

public class Inference2Counts {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		TopicInferencer inferencer = null;
		try {
			inferencer = TopicInferencer.read(new File(args[0]));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//Yes, this is, like, THE case for using StringBuilder, but...
      	String[] outToks = args[0].split("/");
      	String outBase = "";
      	for(int i = 0; i < outToks.length-1; ++i) {
      		outBase += outToks[i] + "/";
      	}
		
		FileOutputStream fout = new FileOutputStream(outBase + "typeTopicCounts.mallet");
		ObjectOutputStream oos = new ObjectOutputStream(fout);
		oos.writeObject(inferencer.getTypeTopicCounts());
		
		fout = new FileOutputStream(outBase + "tokensPerTopic.mallet");
		oos = new ObjectOutputStream(fout);
		oos.writeObject(inferencer.getTokensPerTopic());
	}

}
