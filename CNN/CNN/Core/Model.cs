using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    public class Model
    {
        protected ILayer[] cnn;
        
        public Model() { }

        public Model Configure(string configStr, ILayer input)
        {
            string[] cfg = configStr.Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);

            int n = cfg.Length + 1;
            cnn = new ILayer[n];
            cnn[0] = input;

            for (int i = 0; i < n - 1; i++)
                cnn[i + 1] = new Layer().Configure(cfg[i], cnn[i]);
            //comment
            return this;
        }


        public Model Forward()
        {
            int n = cnn.Length;
            for (int i=1; i<n; i++)
            {
                int size = cnn[i].filters.Count;
                int no = 0;

                if (cnn[i].filters[0].GetType() == typeof(Filters.Connection))
                {
                    //Connection layer
                    for(int l=0; l< cnn[i].Connections.Count; l++)
                    {
                        no++;
                        //Console.WriteLine(no);
                        double sum = 0;
                        for (int m = 0; m < cnn[i].Connections.Count; m++)
                        {
                            IList<Filter> flt = new List<Filter>();
                            var val = cnn[i].Connections[l][m].value[0][0].Value * cnn[i].Connections[l][m].Source.value[0][0].Value;
                            sum += val;
                        }            
                        Actfunc actfunc = new Actfunc();
                        double relu_output;
                        switch (cnn[i].Connections[0][0].Activation)
                        {
                            case "relu":
                                relu_output = actfunc.ReLu(sum + cnn[i].fMaps[l].Bias);
                                cnn[i].fMaps[l].value[0][0].Value = relu_output;
                                Console.WriteLine(cnn[i].fMaps[l].value[0][0].Value);
                                Console.WriteLine(no);
                                break;                      
                        }
                    }
                    if (cnn[i].filters[0].Activation == "softmax")
                    {
                        List<double> output = new List<double>();
                        Console.WriteLine("output layer");
                        for (int s=0; s<cnn[i].fMaps.Count; s++)
                        {
                            output.Add(cnn[i].fMaps[s].value[0][0].Value);
                            Console.WriteLine(cnn[i].fMaps[s].value[0][0].Value);
                        }
                        Actfunc actfunc = new Actfunc();
                        var probabilites =  actfunc.Softmax(output);
                        for (int s = 0; s < cnn[i].fMaps.Count; s++)
                        {
                            cnn[i].fMaps[s].value[0][0].Value = probabilites[s];
                            Console.WriteLine(probabilites[s]);
                        }
                    }
                }
                else
                {
                    //Convolution or pooling layer
                    for (int j = 0; j < size; j++)
                    {
                        Filter k = cnn[i].filters[j].forward();
                    }
                    if ((cnn[i].filters[0].GetType()==typeof(Filters.Maxpool)) && cnn[i + 1].filters[0].GetType()==typeof(Filters.Connection))
                    {
                        if ((cnn[i].fMaps.Count * cnn[i].fMaps[0].Size * cnn[i].fMaps[0].Size) != cnn[i + 1].Connections[0].Count)
                        {
                            throw new Exception("Flatten dimension mismatch");
                        }

                        Linalg linalg = new Linalg();
                        IList<IList<fMap>> feature_maps = new List<IList<fMap>>();
                        for (int l = 0; l < cnn[i].fMaps.Count; l++)
                        {
                            Console.WriteLine("Flattening...");
                            feature_maps.Add(linalg.Flatten(cnn[i].fMaps[l]));
                        }
                        for (int t = 0; t < cnn[i + 1].Connections.Count; t++)
                        {
                            int count = 0;
                            for (int m = 0; m < feature_maps.Count; m++)
                            {
                                for (int h = 0; h < feature_maps[0].Count; h++)
                                {
                                    IList<Filter> flt = new List<Filter>();
                                    flt = cnn[i + 1].Connections[t];
                                    flt[count].Source = feature_maps[m][h];
                                    count++;
                                }

                            }
                        }
                    }
                }

            }
            Console.WriteLine(cnn[4].fMaps[0].value[0][0].Value);
            Console.WriteLine(cnn[4].fMaps[1].value[0][0].Value);
            Console.WriteLine(cnn[4].fMaps[2].value[0][0].Value);
            Console.WriteLine("Output layer");
            Console.WriteLine(cnn[5].fMaps[0].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[1].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[2].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[3].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[4].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[5].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[6].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[7].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[8].value[0][0].Value);
            Console.WriteLine(cnn[5].fMaps[9].value[0][0].Value);



            return this;
        }

        public Model backward(int parameters)
        {
            //backward propagation
            //calculate gradients
            //returns gradients
            return this;
        }

        public int[] train(int num_of_classes, int lr, int beta1, int beta2, int batch_size, int num_epochs)
        {        
            //training loop
            //model = Model.forward(params)
            //gradients = Model.backward(model)
            //loss = Loss.categoricalCrossEntropy(model.output, labels)
            //parameters, cost = optimizer.adamGD(gradients, loss)
            //return optimized parameters and cost
            return null;
        }

        public int [] predict()
        {
            //Make predictions with trained filters and weights
            return null;
        }

        public ILayer[] Layer
        {
            get { return cnn;}
        }
    }
}