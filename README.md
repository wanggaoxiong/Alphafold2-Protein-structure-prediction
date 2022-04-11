# <center> 基于Alphafold2进行蛋白质分子结构预测 </center>
医药行业一直以来都是一个相对封闭的行业，专业性特别强，相关领域知识与其他行业不共通，让医药行业与外界之间始终隔着一道墙。如今这堵墙正在因为数字化技术的介入逐渐瓦解。越来越多人工智能企业，开始与药物研发者们合作，将人工智能技术应用于新药研发的各个环节中，加速新药研发流程。
<img src="https://pic.rmb.bdstatic.com/bjh/down/bb7118770db49c06d20a413981329f13.gif">

## AI+为医药行业带来的价值
医药研发是制药企业价值和生命力的核心所在，但新药研发周期长、成功率低和研发费用高一直是该领域内的三大困境。以深度学习为代表的人工智能技术，凭借其强大的关系发现能力和计算能力加速医药研发流程。
新药研发主要包括药物发现、临床前研究、临床研究以及审批与上市四个阶段。
<div align=center><img src="https://p3.itc.cn/q_70/images03/20201024/a09ad4f89db94fd4aa7ac5c5190239c3.png"></div>
药物发现阶段主要涉及疾病选择、靶点发现和化合物合成。而临床前研究阶段则以化合物筛选、晶型预测、化合物验证为主，包括药物的构效关系分析、稳定性分析、安全性评价和ADMET分析等。
AI主要应用其强大的关系发现能力和计算能力助力新药研发。在计算方面，AI具备的强大认知计算能力，可以对候选化合物进行虚拟筛选，更快的筛选出具有较高活性的化合物，为后期临床试验做准备。

### AI+新药主要应用场景：
<div align=center><img src="http://p5.itc.cn/q_70/images03/20201024/81466e7669014505b2e681316cf376c8.png"></div>

## AlphaFold2的诞生
2018年的CASP 13（国际权威的蛋白质结构预测竞赛，每2年举办一次）上，谷歌DeepMind团队的AlphaFold拿下了70多分，打败众多研究团队，取得人工组第一，在该领域取得了里程碑式的进展。在2020年的CASP 14上，谷歌DeepMind团队的AlphaFold2以惊人的92.4分登顶第一[1]，这一结果也被认为是基本解决了“困扰了生物学家50年”的问题，获得重大突破。92.4分，指的是对竞赛目标蛋白的预测精度GDT_TS分数达到92.4，一般认为该分数超过90分，基本可以替代实验方式啦，这也意味着AlphaFold2预测的结果与实验得到的蛋白质结构基本一致。
2021年7月15日， DeepMind团队在国际顶级期刊《Nature》上发表论文，详细描述了AlphaFold2的设计思路，并提供了可供运行的基于JAX的模型和代码。

## AlphaFold2算法设计思想
AlphaFold2通过独特的神经网络和训练过程设计，第一次端到端地学习蛋白质结构。整个算法框架通过协同学习蛋白质的多序列比对（MSA）和氨基酸对（pairwise）的表征，将蛋白质序列的进化信息、蛋白质结构的物理和几何约束信息结合到深度学习网络中。我们将从数据预处理、Evoformer和Structure Module三个模块分析AlphaFold2算法的设计思想。
<div align=center><img src="https://p2.itc.cn/q_70/images03/20220212/2c496a431965421f8636cabcef148792.png"></div>

- 数据处理

预测蛋白结构时，AlphaFold2会利用氨基酸序列信息在蛋白质库中搜索多序列比对（MSA）。MSA可以反映氨基酸序列中的保守性区域（即不容易产生突变），这些保守性区域和蛋白质的结构息息相关，比如可能被折叠在蛋白质内层，不容易和外界产生相互作用，进而不易受影响发生突变。在AlphaFold2的数据预处理中，为了减少模型运算量，会先对MSA中的序列进行聚类，取每个类别中心的序列作为main MSA特征。除了MSA，AlphaFold2的另一个重要输入是氨基酸对（pairwise）的特征。作为main MSA的补充，Alphafold2会随机采样非聚类中心的序列作为extra MSA输入一个4层的网络提取pairwise特征，然后和模版提取的pairwise特征相加后得到最终pairwise特征。main MSA特征和pairwise特征通过48层Evoformer进行表征融合。
- Evoformer

Evoformer网络的设计动机是想利用Self-Attention机制学习蛋白质的三角几何约束信息，同时让MSA表征带来的共进化信息和pairwise表征的结构约束信息相互影响，使得模型能直接推理出空间信息和进化信息的联系。
- Stucture Module

Structure Module承担着把Evoformer得到的表征解码成蛋白质中每个重原子(C,N,O,S)坐标的任务。为了简化从神经网络预测值到原子坐标的转换，AlphaFold2结合蛋白质中20类氨基酸的结构特性，将重原子分成不同二面角转角决定的组，这样就可以根据给定的起始位置，利用二面角和氨基酸已知的键长键角信息解码出原子坐标。这种结构编码方法相比直接预测坐标(x,y,z)大大降低了神经网络的预测空间，使得端到端结构学习成为可能。
<div align=center><img src="https://p9.itc.cn/q_70/images03/20220212/3cde3ea89c054ca6a4cebe24431b2328.png"></div>
赖氨酸的转角编码方式示例：蓝色平面（C,Cα,Cβ）确定后，根据预测的蓝色-紫色平面的二面角χ1和已知的C-C键长，Cγ-Cβ-N键角即可确定Cγ的空间坐标，重复类似步骤，可以得到Cδ,Cε, N等重原子坐标。

## Alphafold2安装使用
如下将基于AMZLinux2环境安装部署Alphafold2，更多细节请参考：https://github.com/deepmind/alphafold
- 安装docker
```
sudo amazon-linux-extras install docker
sudo systemctl --now enable docker
sudo docker run --rm hello-world
docker –version
sudo chkconfig docker on
docker pull centos
sudo docker pull centos
sudo docker images centos
sudo docker run -i -t centos /bin/bash
exit
```
- 配置docker非root用户执行
参考链接：https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user
```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```
- 安装NVIDA驱动（并给ec2相关role）
参考链接：https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-driver-instance-type
```
sudo yum update –y
sudo reboot
sudo yum install -y gcc kernel-devel-$(uname -r)
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
aws s3 ls --recursive s3://ec2-linux-nvidia-drivers/
chmod +x NVIDIA-Linux-x86_64*.run
sudo /bin/sh ./NVIDIA-Linux-x86_64*.run
If you are using Amazon Linux 2 with kernel version 5.10, use the following command to install the GRID driver.
sudo CC=/usr/bin/gcc10-cc ./NVIDIA-Linux-x86_64*.run
sudo reboot
```
- 安装配置和测试NVIDIA Container Toolkit
参考链接：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```
sudo yum install nvidia-container-toolkit –y
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum clean expire-cache
sudo systemctl restart docker
sudo systemctl enable docker
(service docker status)
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

```
- 安装aria2c和rsync工具
```
sudo amazon-linux-extras install epel
yum install aria2 –y
```
- 下载alphafold2数据库
```
  yum install git –y
git clone https://github.com/deepmind/alphafold.git
进入script目录通过脚本cp所有数据
scripts/download_all_data.sh <DOWNLOAD_DIR>
然后对每个文件进行解压，目录结构如下：
$DOWNLOAD_DIR/                             # Total: ~ 2.2 TB (download: 438 GB)
    bfd/                                   # ~ 1.7 TB (download: 271.6 GB)
        # 6 files.
    mgnify/                                # ~ 64 GB (download: 32.9 GB)
        mgy_clusters_2018_12.fa
    params/                                # ~ 3.5 GB (download: 3.5 GB)
        # 5 CASP14 models,
        # 5 pTM models,
        # 5 AlphaFold-Multimer models,
        # LICENSE,
        # = 16 files.
    pdb70/                                 # ~ 56 GB (download: 19.5 GB)
        # 9 files.
    pdb_mmcif/                             # ~ 206 GB (download: 46 GB)
        mmcif_files/
            # About 180,000 .cif files.
        obsolete.dat
    pdb_seqres/                            # ~ 0.2 GB (download: 0.2 GB)
        pdb_seqres.txt
    small_bfd/                             # ~ 17 GB (download: 9.6 GB)
        bfd-first_non_consensus_sequences.fasta
    uniclust30/                            # ~ 86 GB (download: 24.9 GB)
        uniclust30_2018_08/
            # 13 files.
    uniprot/                               # ~ 98.3 GB (download: 49 GB)
        uniprot.fasta
    uniref90/                              # ~ 58 GB (download: 29.7 GB)
        uniref90.fasta

```

