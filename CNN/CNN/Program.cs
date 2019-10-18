using System;
using CNN.Core;
using NN.Core;
using System.Collections.Generic;
using System.Linq;

namespace CNN
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            MainClass mainClass = new MainClass();
            //mainClass.random_gaussian_test();
            //mainClass.max_test();
            mainClass.model_test();
        }

        public void max_test()
        {
            Linalg linalg = new Linalg();
            Filter m = new Filter(null, 4, 2);
            m.value[0][0] = new Node<double>(3.0); m.value[0][1] = new Node<double>(7.0); m.value[0][2] = new Node<double>(2.0); m.value[0][3] = new Node<double>(5.0);
            m.value[1][0] = new Node<double>(1.0); m.value[1][1] = new Node<double>(8.0); m.value[1][2] = new Node<double>(4.0); m.value[1][3] = new Node<double>(2.0);
            m.value[2][0] = new Node<double>(2.0); m.value[2][1] = new Node<double>(1.0); m.value[2][2] = new Node<double>(9.0); m.value[2][3] = new Node<double>(3.0);
            m.value[3][0] = new Node<double>(15.0); m.value[3][1] = new Node<double>(4.0); m.value[3][2] = new Node<double>(7.0); m.value[3][3] = new Node<double>(1.0);
            var max = linalg.Max(m);
            Console.WriteLine(max.Value);

        }

        public void model_test()
        {
            string configStr = "";
            // configStr += "image(size=256);";
            configStr += "conv(size=5, depth=8, stride=1, fmapsize=24, bias=0, activation=relu); ";
            configStr += "conv(size=5, depth=8, stride=1, fmapsize=20, bias=0, activation=relu); ";
            configStr += "pool(size=2, depth=8, stride=2, fmapsize=10, bias=0, activation=None); ";
            configStr += "conn(size=1, depth=128, stride=2, fmapsize=1, bias=0, activation=relu);";
            configStr += "conn(size=1, depth=10, stride=2, fmapsize=1, bias=0, activation=softmax);";
            Image image = new Image().Configure(28, 28);
            CNN.Core.Model model = new CNN.Core.Model().Configure(configStr, image);
            model.Forward();
            //calculate loss
            //model.backward();
            //Console.WriteLine(model.Layer[2].ToString());
        }
        public void random_gaussian_test()
        {
            Random rand = new Random(); //reuse this if you are generating many
            double mean = 0;
            double stdDev = 0.047855339;
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + stdDev * randStdNormal;
            Console.WriteLine(randStdNormal);
            Console.WriteLine(randNormal);
        }

        public List<double> Softmax_test(List<double> output)
        {
            List<double> vs = new List<double>();
            List<double> softmax = new List<double>();

            for (int i = 0; i < output.Count; i++)
            {
                vs.Add(Math.Exp(output[i]));
            }
            var total = vs.Sum();
            for (int i = 0; i < vs.Count; i++)
            {
                softmax.Add(vs[i] / total);
            }
            return softmax;

            //List<double> X = new List<double>();
            //X.Add(-0.00014549);
            //X.Add(-0.00120929);
            //X.Add(0.00356441);
            //X.Add(0.00181518);
            //X.Add(0.00325538);
            //X.Add(-0.00260445);
            //X.Add(0.00170222);
            //X.Add(0.0007031);
            //X.Add(-0.00151826);
            //X.Add(-0.00176412);
            //var r = mainClass.Softmax_test(X);
            //Console.WriteLine(r[1]);
        }
    }
}
