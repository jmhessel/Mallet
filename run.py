import subprocess, os
call = lambda x: subprocess.call(x, shell = True)

if not os.path.exists("dmr.maxentmodel"):
    call("bin/mallet run cc.mallet.topics.tui.DMRLoader kickstarter_text_small.txt kickstarter_features_small.txt kickstarter_instances.mallet")
    call("bin/mallet run cc.mallet.topics.DMRTopicModel kickstarter_instances.mallet 30")

call("bin/mallet train-topics --num-iterations 0 --input-state dmrState.state --inferencer-filename myInferencer.mallet --input kickstarter_instances.mallet --num-topics 30")

call("bin/mallet run cc.mallet.topics.tui.Inference2Counts myInferencer.mallet")

call("bin/mallet run cc.mallet.topics.DMRTopicInferencer kickstarter_instances.mallet dmr.maxentmodel typeTopicCounts.mallet tokensPerTopic.mallet dmr.oldcounts")
