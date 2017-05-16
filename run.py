import subprocess, os, numpy as np
call = lambda x: subprocess.call(x, shell = True)
n_rep = 10

if not os.path.exists("dmr.maxentmodel"):
    call("bin/mallet run cc.mallet.topics.tui.DMRLoader kickstarter_text.small kickstarter_features.small kickstarter_instances.mallet 1000")
    call("bin/mallet run cc.mallet.topics.DMRTopicModel kickstarter_instances.mallet-training 30")

call("bin/mallet train-topics --num-iterations 0 --input-state dmrState.state --inferencer-filename myInferencer.mallet --input kickstarter_instances.mallet-training --num-topics 30")

call("bin/mallet run cc.mallet.topics.tui.Inference2Counts myInferencer.mallet")
call('rm -rf /tmp/likelihood_files')
os.makedirs("/tmp/likelihood_files")
for i in range(n_rep):
    call("bin/mallet run cc.mallet.topics.DMRTopicInferencer kickstarter_instances.mallet-testing dmr.maxentmodel typeTopicCounts.mallet tokensPerTopic.mallet dmr.oldcounts")
    call("cp likelihood.txt /tmp/likelihood_files/{}.txt".format(i))

def load_arr(fname):
    vals = []
    with open(fname) as f:
        for line in f:
            vals.append(float(line))
    return np.array(vals)

arrs = np.hstack([np.expand_dims(load_arr('/tmp/likelihood_files/{}.txt'.format(i)), 1) for i in range(n_rep)])
means = np.mean(arrs, axis = 1)
import statsmodels.stats.api as sms

for i in range(len(means)):
    print(sms.DescrStatsW(arrs[i,:]).tconfint_mean())
