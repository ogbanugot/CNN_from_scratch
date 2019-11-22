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
            Console.WriteLine("Configuring model....");
            string[] cfg = configStr.Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);

            int n = cfg.Length + 1;
            cnn = new ILayer[n];
            cnn[0] = input;

            for (int i = 0; i < n - 1; i++)
                cnn[i + 1] = new Layer().Configure(cfg[i], cnn[i]);
            //comment
            Console.WriteLine("Configuration complete.");

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
                        double sum = 0;
                        for (int m = 0; m < cnn[i].Connections.Count; m++)
                        {
                            IList<Filter> flt = new List<Filter>();
                            var val = cnn[i].Connections[l][m].value[0][0].Value * cnn[i].Connections[l][m].Source.value[0][0].Value;
                            sum += val;
                        }            
                        Actfunc actfunc = new Actfunc();
                        double induced_field = sum + cnn[i].fMaps[l].Bias[0];
                        cnn[i].fMaps[l].InducedField[0][0] = induced_field;
                        double relu_output;
                        switch (cnn[i].Connections[0][0].Activation)
                        {
                            case "relu":
                                relu_output = actfunc.ReLu(induced_field);
                                cnn[i].fMaps[l].value[0][0].Value = relu_output;
                                
                                break;                      
                        }
                    }
                    if (cnn[i].filters[0].Activation == "softmax")
                    {
                        List<double> output = new List<double>();
                        for (int s=0; s<cnn[i].fMaps.Count; s++)
                        {
                            output.Add(cnn[i].fMaps[s].InducedField[0][0]);
                        }
                        Actfunc actfunc = new Actfunc();
                      
                        var probabilites =  actfunc.Softmax(output);
                        for (int s = 0; s < cnn[i].fMaps.Count; s++)
                        {
                            cnn[i].fMaps[s].value[0][0].Value = probabilites[s];
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
            //Console.WriteLine("\n");
            //Console.WriteLine(cnn[4].fMaps[0].InducedField[0][0]);
            //Console.WriteLine(cnn[4].fMaps[1].InducedField[0][0]);
            //Console.WriteLine(cnn[4].fMaps[2].InducedField[0][0]);
            //Console.WriteLine("\n");
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
            Console.WriteLine("\n\n");

            return this;
        }

        public Model Backward(double []label)
        {
            Console.WriteLine("\n\n");
            Console.WriteLine("backprop");
            int n = cnn.Length;
            for (int i = n-1; i >= 1; i--)
            {
                int size = cnn[i].filters.Count;
                //int no = 0;
                //output layer
                if (i==n-1 && cnn[i].filters[0].GetType() == typeof(Filters.Connection))
                {
                    Console.WriteLine("output layer");
                    //for each fmap
                    for (int j=0; j<cnn[i].fMaps.Count; j++)
                    {
                        //dout = prob - label
                        cnn[i].fMaps[j].Gradient[0][0] = cnn[i].fMaps[j].value[0][0].Value - label[j];
                        //bias = dout
                        cnn[i].fMaps[j].Bias[1] = cnn[i].fMaps[j].Gradient[0][0];
                    }
                    //derivative of weights
                    //for all connections
                    for (int h=0; h<cnn[i].Connections.Count; h++)
                    {
                        //for each weight in connection
                        for(int m=0; m<cnn[i].Connections[0].Count; m++)
                        {
                            //dw = previous_layer_output * gradient__of_current_layer_output
                            cnn[i].Connections[h][m].Gradient[0][0] = cnn[i].Connections[h][m].Source.value[0][0].Value * cnn[i].fMaps[h].Gradient[0][0];
                        }
                    }

                }

                if((i != n - 1) && (cnn[i].filters[0].GetType() == typeof(Filters.Connection)))
                {
                    Console.WriteLine("hidden layer");
                    //hidden layer
                    Actfunc actfunc = new Actfunc();
                    double function_derivative = 0;
                    double sum = 0;

                    //for each fmap in hidden layer
                    for (int j=0; j<cnn[i].fMaps.Count; j++)
                    {
                        switch (cnn[i].filters[i].Activation)
                        {
                            case "relu":
                                function_derivative = actfunc.DReLuConn(cnn[i].fMaps[i].InducedField[0][0]);
                                break;
                        }
                        //for each fmap in proceeding layer
                        for (int h=0; h<cnn[i+1].fMaps.Count; h++)
                        {
                            //for each fmap and its corresponding weight
                            sum += (cnn[i + 1].Connections[h][j].value[0][0].Value) * (cnn[i + 1].fMaps[h].Gradient[0][0]);
                        }
                        cnn[i].fMaps[j].Gradient[0][0] = sum * function_derivative;
                        //bias
                        cnn[i].fMaps[j].Bias[1] = cnn[i].fMaps[j].Gradient[0][0];

                    }

                    //derivative of weights
                    //for all connections
                    for (int h = 0; h < cnn[i].Connections.Count; h++)
                    {
                        //for each weight in connection
                        for (int m = 0; m < cnn[i].Connections[0].Count; m++)
                        {
                            //dw = previous_layer_output (source) * gradient__of_current_layer_output
                            cnn[i].Connections[h][m].Gradient[0][0] = cnn[i].Connections[h][m].Source.value[0][0].Value * cnn[i].fMaps[h].Gradient[0][0];
                        }
                    }
                }

                if ((i != n - 1) && (cnn[i].filters[0].GetType() == typeof(Filters.Maxpool)) && (cnn[i+1].filters[0].GetType() == typeof(Filters.Connection)))
                {
                    Console.WriteLine("pool layer");
                    // derivative of Pooled layer
                    double sum =0;
                    int number_of_fmaps = cnn[i].fMaps.Count;
                    int size_of_fmaps = cnn[i].fMaps[0].value.Length;
                    int number_of_filters = size_of_fmaps * size_of_fmaps * number_of_fmaps;
                    double[] pool_gradient = new double[number_of_filters];
                    for (int j = 0; j < number_of_filters; j++)
                    {
                        //for each fmap in proceeding layer
                        for (int h = 0; h < cnn[i + 1].fMaps.Count; h++)
                        {
                            //for each fmap and its corresponding weight
                            sum += (cnn[i + 1].Connections[h][j].value[0][0].Value) * (cnn[i + 1].fMaps[h].Gradient[0][0]);
                        }
                        pool_gradient[j] = sum;
                    }
                    int index = 0;
                    //to assign gradient to pooled maps
                    for (int l=0; l<number_of_fmaps; l++)
                    {
                        for(int m=0; m<size_of_fmaps; m++)
                        {
                            for (int p = 0; p < size_of_fmaps; p++)
                            {
                                cnn[i].fMaps[l].Gradient[m][p] = pool_gradient[index];
                                index++;
                            }
                        }
                    }
                }

                if ((i != n - 1) && (cnn[i].filters[0].GetType() == typeof(Filters.Convolution)) && (cnn[i + 1].filters[0].GetType() == typeof(Filters.Maxpool)))
                {
                    //maxpool to convolution backprop
                    Console.WriteLine("convolution layer");
                    for (int j=0; j<cnn[i].fMaps.Count; j++)
                    {
                        cnn[i].filters[j].Maxbackward(cnn[i + 1].fMaps[j], cnn[i+1].filters[i].Stride);
                    }
                }

                if ((i != n - 1) && (cnn[i].filters[0].GetType() == typeof(Filters.Convolution)) && (cnn[i + 1].filters[0].GetType() == typeof(Filters.Convolution)))
                {
                    //convolution to convolution backprop
                    Console.WriteLine("convolution layer");
                    for (int j = 0; j < cnn[i+1].fMaps.Count; j++)
                    {
                        cnn[i].filters[j].Backward(cnn[i+1].fMaps[j], cnn[i].filters[i].Activation);

                    }
                }

            }

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