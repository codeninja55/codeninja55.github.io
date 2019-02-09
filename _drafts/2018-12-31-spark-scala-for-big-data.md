---
layout: post
title: Installing Spark 2.1.3
status: draft
---

While I mostly prefer to do my data science projects in Python, as part of my research assistant work I am 
required to use Spark and Scala to write application that can handle big data distributed across multiple clusters 
of computers. I am still very new to Scala and Spark, however, I have learnt quickly enough to note down some tips and 
an installation guide. 

In this post, I am going to note the steps used to install [Apache Spark](https://spark.apache.org/) and all of its associated dependencies.

So let's begin. 


## Installing Oracle Java 8 and 11 on Ubuntu 18.04

### Java 8

```bash
$ java -version
$ sudo apt update && sudo apt upgrade
$ sudo add-apt-repository -y ppa:webupd8team/java
$ actionMessage "Installing Oracle Java 8 JRE and JDK"
$ echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections
$ echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections

$ sudo apt install -y oracle-java8-installer
$ sudo apt install -y default-jre
$ sudo apt install -y default-jdk
```

Normally, we would set Java 8 as the default Java. However, as this guide will install Java 11 in the next section, we will avoid setting it as default.

If you would like to use Java 8 as your default though, you can also run the following command `sudo apt install -y oracle-java8-set-default` which will easily configure your Ubuntu to run Java 8 for `java`, `javac`, and `jar`. Please note this will often override any configurations you manually make using `sudo update-alternatives` so if you want better control, I would recommend not using this command. 

### Java 11

While Spark 2.1.3 is best used with Java 8, my Ubuntu also uses Java for other things and it is fairly easy to switch
 between them in Linux. So for this reason, I like to have the latest Java available and only use Java 8 for any 
 Spark-specific purposes. 

There are two ways to install Oracle Java in Ubuntu. I will explain both and you can decide which you prefer. In my 
opinion, I prefer the second method as it allows me more control over where I can install Java.

#### Option 1: Using Ubuntu package manager

```bash
$ sudo add-apt-repository ppa:linuxuprising/java
$ sudo apt update
$ echo oracle-java11-installer shared/accepted-oracle-license-v1-2 select true | sudo /usr/bin/debconf-set-selections
$ sudo apt install oracle-java11-installer
$ sudo apt install oracle-java11-set-default
```



#### Option 2: Installing from source

Go to [here](https://www.oracle.com/technetwork/java/javase/downloads/index.html) to get the link to the latest version of Java 11. 

![Oracle Java Download Page](/images/2018-12-31-java_download-1.png)

After clicking the _Download_ button, you will need to accept the license. Then right click on the **jdk-11.0.1_linux-x64_bin.deb** link and select _Copy link address_ (Chrome) or _Copy Link Location_ (Firefox). Then run the following commands to download and install Oracle Java 11, replacing the web URL with the one you copied. 

```bash
$ wget --no-cookies --no-check-certificate --header "Cookie: oraclelicense=accept-securebackup-cookie" \
https://download.oracle.com/otn-pub/java/jdk/11.0.1+13/90cf5d8f270a4347a95050320eef3fb7/jdk-11.0.1_linux-x64_bin.deb \ 
$ sudo dpkg -i jdk-*.deb && rm jdk*.deb 
```

Run the following to check where your new Java directory is installed. 

```bash
$ ls /usr/lib/jvm/

default-java  java-1.11.0-openjdk-amd64  java-11-openjdk-amd64  java-8-oracle  jdk-11.0.1
```

As you can see, I have multiple versions of Java installed. The one I want as default will be in the `jdk-11.0.1` directory which I will to add as a configuration alternative next. 

**Updating System Default Java Configurations**

We want to configure the Ubuntu system to use the latest Java we installed. But before we can do that, we must install our newly installed version to `update-alternatives --config` by running the following:

```bash
$ sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.1/bin/java 1102
$ sudo update-alternatives --config java

There are 3 choices for the alternative java (providing /usr/bin/java).

  Selection    Path                                         Priority   Status
------------------------------------------------------------
  0            /usr/lib/jvm/jdk-11.0.1/bin/java              1102      auto mode
  1            /usr/lib/jvm/java-11-openjdk-amd64/bin/java   1101      manual mode
* 2            /usr/lib/jvm/java-8-oracle/jre/bin/java       1081      manual mode
  3            /usr/lib/jvm/jdk-11.0.1/bin/java              1102      manual mode

Press <enter> to keep the current choice[*], or type selection number: 3
```

From here you want to input `3` to set `jdk-11.0.1` as the default. 

Next, we want to make Java 11 the default Java compiler for Ubuntu. It is also very important you replace `jdk-11.0.1` with your installation directory from above. 

```bash
$ sudo update-alternatives --install /usr/bin/jar jar /usr/lib/jvm/jdk-11.0.1/bin/jar 1102
$ sudo update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/jdk-11.0.1/bin/javac 1102
$ sudo update-alternatives --set jar /usr/lib/jvm/jdk-11.0.1/bin/jar
$ sudo update-alternatives --set javac /usr/lib/jvm/jdk-11.0.1/bin/javac
```

Running the following commands will confirm you have made the appropriate changes and if you ever need to change back to another version of Java.

```bash
$ sudo update-alternatives --config javac

There are 3 choices for the alternative javac (providing /usr/bin/javac).

  Selection    Path                                          Priority   Status
------------------------------------------------------------
  0            /usr/lib/jvm/jdk-11.0.1/bin/javac              1102      auto mode
  1            /usr/lib/jvm/java-11-openjdk-amd64/bin/javac   1101      manual mode
  2            /usr/lib/jvm/java-8-oracle/bin/javac           1081      manual mode
* 3            /usr/lib/jvm/jdk-11.0.1/bin/javac              1102      manual mode

Press <enter> to keep the current choice[*], or type selection number: 

$ sudo update-alternatives --config jar

There are 3 choices for the alternative jar (providing /usr/bin/jar).

  Selection    Path                                        Priority   Status
------------------------------------------------------------
  0            /usr/lib/jvm/jdk-11.0.1/bin/jar              1102      auto mode
  1            /usr/lib/jvm/java-11-openjdk-amd64/bin/jar   1101      manual mode
  2            /usr/lib/jvm/java-8-oracle/bin/jar           1081      manual mode
* 3            /usr/lib/jvm/jdk-11.0.1/bin/jar              1102      manual mode

Press <enter> to keep the current choice[*], or type selection number: 
```

Finally, we want to ensure Ubuntu recognises the correct version of Java 11 so we will check a few things. Ensure your output is similar to mine based on the version you decided to install. 

```bash
$ java -version
java version "11.0.1" 2018-10-16 LTS
Java(TM) SE Runtime Environment 18.9 (build 11.0.1+13-LTS)
Java HotSpot(TM) 64-Bit Server VM 18.9 (build 11.0.1+13-LTS, mixed mode)

$ javac -version
javac 11.0.1

$ jar --version
jar 11.0.1
```



## Installing Scala 2.11.12

https://www.scala-lang.org/download/2.11.12.html

The latest Scala version as of this post is 2.12.1. However, Spark 2.1.3 works best with Scala 2.11.12 so we will install from a Debian package. Follow the following steps to download and install Scala 2.11.12.

```bash
$ sudo apt purge scala-library scala
$ wget -c https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.deb -O $HOME/Downloads/scala-2.11.12.deb --read-timeout=5 --tries=0
$ sudo dpkg -i $HOME/Downloads/2.11.12/scala-2.11.12.deb
$ sudo apt update && sudo apt -y upgrade
```

This will install Scala 2.11 to `/usr/share/scala-2.11/bin/scala`.



## Installing simple build tool (sbt)

https://www.scala-sbt.org/download.html?_ga=2.68279761.1793440311.1546231938-953192740.1546231938

```bash
$ echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
$ sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
$ sudo apt update && sudo apt install sbt
```



## Installing Spark



```bash
$ tar -xvf ${spark_intermed_dir} -C ${spark_intermed_dir}
$ sudo mv ${spark_intermed_dir} /usr/local/
$ sudo ln -s ${spark_dir} /usr/local/spark
$ echo "##### SPARK #####" >> $HOME/.bashrc
$ echo "export SCALA_HOME=/usr/share/scala-2.11" >> $HOME/.bashrc
$ echo "export SPARK_HOME=/usr/local/spark" >> $HOME/.bashrc
$ echo "export PATH=\$SPARK_HOME/bin:\$SCALA_HOME/bin:\$PATH" >> $HOME/.bashrc
```






## Setting up JetBrains IntelliJ IDEA Integrated Development Environment (Optional)





## Using simple built tool from terminal (Optional)

