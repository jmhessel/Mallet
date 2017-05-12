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
			inferencer = TopicInferencer.read(new File("myInferencer.mallet"));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		FileOutputStream fout = new FileOutputStream("typeTopicCounts.mallet");
		ObjectOutputStream oos = new ObjectOutputStream(fout);
		oos.writeObject(inferencer.getTypeTopicCounts());
		
		fout = new FileOutputStream("tokensPerTopic.mallet");
		oos = new ObjectOutputStream(fout);
		oos.writeObject(inferencer.getTokensPerTopic());

	}

}
