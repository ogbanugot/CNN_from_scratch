using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    public class Model
    {
        private string formattingString = "0.0";
        protected ILayer[] cnn;
        double cost;
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
            ////comment
            //Console.WriteLine("Configuration complete.");
            //string g3 = ToGString();
            //System.IO.File.WriteAllText("/home/ugot/mnist/gstring", g3);
            //string g5 = ToGradientStoreString();
            //System.IO.File.WriteAllText("/home/ugot/mnist/grdstring", g5);
            return this;
        }


        public Model Forward(double[][] image)
        {
            int n = cnn.Length;
            for (int i=0; i<n; i++)
            {
                //Image layer add image values 
                if (i == 0)
                {
                    if (cnn[i].fMaps[0].value.Count() != image.Count())
                        throw new Exception("Image size mismatch");

                    for(int l=0; l<cnn[i].fMaps[0].value.Count(); l++)
                        for (int m=0; m<image[0].Count(); m++)
                        {
                            cnn[i].fMaps[0].value[l][m].Value = image[l][m];
                            
                        }
                }
                else
                {
                    if (cnn[i].filters[0].GetType() == typeof(Filters.Connection))
                    {
                        //Connection layer
                        for (int l = 0; l < cnn[i].Connections.Count; l++)
                        {
                            double sum = 0;
                            for (int m = 0; m < cnn[i].Connections[0].Count; m++)
                            {
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
                            for (int s = 0; s < cnn[i].fMaps.Count; s++)
                            {
                                output.Add(cnn[i].fMaps[s].InducedField[0][0]);
                            }
                            Actfunc actfunc = new Actfunc();

                            var probabilites = actfunc.Softmax(output);
                            for (int s = 0; s < cnn[i].fMaps.Count; s++)
                            {
                                cnn[i].fMaps[s].value[0][0].Value = probabilites[s];
                            }
                        }
                    }
                    else
                    {
                        int size = cnn[i].filters.Count;
                        //Convolution or pooling layer
                        for (int j = 0; j < size; j++)
                        {
                            Filter k = cnn[i].filters[j].forward();
                        }
                        if ((cnn[i].filters[0].GetType() == typeof(Filters.Maxpool)) && cnn[i + 1].filters[0].GetType() == typeof(Filters.Connection))
                        {
                            if ((cnn[i].fMaps.Count * cnn[i].fMaps[0].Size * cnn[i].fMaps[0].Size) != cnn[i + 1].Connections[0].Count)
                            {
                                throw new Exception("Flatten dimension mismatch");
                            }

                            Linalg linalg = new Linalg();
                            IList<IList<fMap>> feature_maps = new List<IList<fMap>>();
                            for (int l = 0; l < cnn[i].fMaps.Count; l++)
                            {
                                feature_maps.Add(linalg.Flatten(cnn[i].fMaps[l]));
                            }
                            for (int t = 0; t < cnn[i + 1].Connections.Count; t++)
                            {
                                int count = 0;
                                for (int m = 0; m < feature_maps.Count; m++)
                                {
                                    for (int h = 0; h < feature_maps[0].Count; h++)
                                    {

                                        cnn[i + 1].Connections[t][count].Source = feature_maps[m][h];
                                        count++;
                                    }

                                }
                            }
                        }
                    }


                }
                }
            return this;
        }

        public Model Backward(double [][]label)
        {

            int n = cnn.Length;
            for (int i = n-1; i >= 1; i--)
            {
                int size = cnn[i].filters.Count;
                //int no = 0;
                //output layer
                if (i==n-1 && cnn[i].filters[0].GetType() == typeof(Filters.Connection))
                {
                    //for each fmap
                    for (int j=0; j<cnn[i].fMaps.Count; j++)
                    {
                        //dout = prob - label
                        cnn[i].fMaps[j].Gradient[0][0] = cnn[i].fMaps[j].value[0][0].Value - label[0][j];
                        //bias = dout
                        cnn[i].fMaps[j].Bias[1] = cnn[i].fMaps[j].Gradient[0][0];
                        cnn[i].fMaps[j].Bias[2] += cnn[i].fMaps[j].Gradient[0][0];

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
                            cnn[i].Connections[h][m].GradientStorage[0][0] += cnn[i].Connections[h][m].Source.value[0][0].Value * cnn[i].fMaps[h].Gradient[0][0];

                        }
                    }

                }

                if((i != n - 1) && (cnn[i].filters[0].GetType() == typeof(Filters.Connection)))
                {
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
                                function_derivative = actfunc.DReLuConn(cnn[i].fMaps[j].InducedField[0][0]);
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
                        cnn[i].fMaps[j].Bias[2] += cnn[i].fMaps[j].Gradient[0][0];


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
                            cnn[i].Connections[h][m].GradientStorage[0][0] += cnn[i].Connections[h][m].Source.value[0][0].Value * cnn[i].fMaps[h].Gradient[0][0];

                        }
                    }
                }

                if ((i != n - 1) && (cnn[i].filters[0].GetType() == typeof(Filters.Maxpool)) && (cnn[i+1].filters[0].GetType() == typeof(Filters.Connection)))
                {
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
                    for (int j=0; j<cnn[i].fMaps.Count; j++)
                    {
                        cnn[i].filters[j].Maxbackward(cnn[i + 1].fMaps[j], cnn[i+1].filters[j].Stride);
                    }
                }

                if ((i != n - 1) && (cnn[i].filters[0].GetType() == typeof(Filters.Convolution)) && (cnn[i + 1].filters[0].GetType() == typeof(Filters.Convolution)))
                {
                    //convolution to convolution backprop
                    for (int j = 0; j < cnn[i+1].fMaps.Count; j++)
                    {
                        cnn[i].filters[j].Backward(cnn[i+1].fMaps[j], cnn[i].filters[j].Activation);

                    }
                }

            }

            return this;
        }

        public Model Train(List<List<double[][]>> dataset, double lr, double beta1, double beta2, int batch_size, int num_epochs)
        {
            //training loop
            //for each epoch
            //create batches from dataset (randomize other, perhaps)
            //For each batch in batches
            //Initialize AdamGD(pass model, and batch to it with other params)
            //display most recent cost from Adam
            List<List<double[][]>> batch_image = new List<List<double[][]>>();
            List<List<double[][]>> batch_label = new List<List<double[][]>>();
            Console.WriteLine("Preparing batches....");
            int index = 0;
            while (index < dataset[0].Count)
            {
                int count = dataset[0].Count - index > batch_size ? batch_size : dataset[0].Count - index;
                batch_image.Add(dataset[0].GetRange(index, count));
                batch_label.Add(dataset[1].GetRange(index, count));
                index += batch_size;
            }
        Console.WriteLine("Done preparing batches, starting training loop....");
            //Initialize Optimizer
            Optimizer optimizer = new Optimizer();
            List<double> costHistory = new List<double>();
            //training loop
        for(int epoch=0; epoch<num_epochs; epoch++)
            {
                //Take it batch by batch
                for(int i=0; i<batch_image.Count(); i++)
                {
                   optimizer.AdamGD(batch_image[i], batch_label[i], this, lr, beta1, beta2, batch_size);
                   Console.WriteLine("Epoch {0}/Batch {1}: Loss: {2}", epoch, i, cost);
                }
            }
            costHistory.Add(cost);
            return this;
        }

        public int[] predict()
        {
            //Make predictions with trained filters and weights
            return null;
        }

        public override string ToString()
        {
            string s = "";
            int n = cnn.Length;
            for (int i=0; i<n; i++)
            {
                if (i == 0)
                {
                    s += "\nImage Layer[" + i.ToString(formattingString) + "]\n";
                    s += "\nFmap\n";
                    s += cnn[i].fMaps[0].ToString();
                
                }
                else
                {
                    if (cnn[i].filters[0].GetType() == typeof(Filters.Connection))
                    {
                        //Connection layer
                        s += "\n\n";
                        s += "\nConnected layer[" + i.ToString(formattingString) + "]\n";
                        for (int l = 0; l < cnn[i].Connections.Count; l++)
                        {
                            s += "\nFmap [" + l.ToString(formattingString) + "]\n";
                            s += cnn[i].fMaps[l].ToString();
                            s += "\n";
                            s += "\nConnections[" + l.ToString(formattingString) + "]\n";
                            for (int m = 0; m < cnn[i].Connections[0].Count; m++)
                            {
                                s += "\n";
                                s += cnn[i].Connections[l][m].ToString();
                                s += "\nConnection_Fmap[" + l.ToString(formattingString) + "]";
                                s += cnn[i].Connections[l][m].Source.ToString();

                            }
                        }
                    }
                    else
                    {
                        int size = cnn[i].filters.Count;
                        //Convolution or pooling layer
                        if (cnn[i].filters[0].GetType() == typeof(Filters.Maxpool))
                        {
                            s += "\n\n";
                            s += "\nPooling layer[" + i.ToString(formattingString) + "]\n";
                            for (int j = 0; j < size; j++)
                            {
                                s += "\nFmap [" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].fMaps[j].ToString();
                                s += "\n";
                                s += "\nFilter[" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].filters[j].ToString();
                            }
                        }
                        else
                        {
                            s += "\n\n";
                            s += "\nConvolution layer[" + i.ToString(formattingString) + "]\n";
                            for (int j = 0; j < size; j++)
                            {
                                s += "\nFmap [" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].fMaps[j].ToString();
                                s += "\n";
                                s += "\nFilter[" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].filters[j].ToString();
                            }
                        }
                    }
                }
            }
            return s;
        }

        public string ToGString()
        {
            string s = "";
            int n = cnn.Length;
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    s += "\nImage Layer[" + i.ToString(formattingString) + "]\n";
                    s += "\nFmap\n";
                    s += cnn[i].fMaps[0].ToString();

                }
                else
                {
                    if (cnn[i].filters[0].GetType() == typeof(Filters.Connection))
                    {
                        //Connection layer
                        s += "\n\n";
                        s += "\nConnected layer[" + i.ToString(formattingString) + "]\n";
                        for (int l = 0; l < cnn[i].Connections.Count; l++)
                        {
                            s += "\nFmap [" + l.ToString(formattingString) + "]\n";
                            s += cnn[i].fMaps[l].ToGString();
                            s += "\n";
                            s += "\nConnections[" + l.ToString(formattingString) + "]\n";
                            for (int m = 0; m < cnn[i].Connections[0].Count; m++)
                            {
                                s += cnn[i].Connections[l][m].ToGString();
                            }
                        }
                    }
                    else
                    {
                        int size = cnn[i].filters.Count;
                        //Convolution or pooling layer
                        if (cnn[i].filters[0].GetType() == typeof(Filters.Maxpool))
                        {
                            s += "\n\n";
                            s += "\nPooling layer[" + i.ToString(formattingString) + "]\n";
                            for (int j = 0; j < size; j++)
                            {
                                s += "\nFmap [" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].fMaps[j].ToGString();
                                s += "\n";
                                s += "\nFilter[" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].filters[j].ToGString();
                            }
                        }
                        else
                        {
                            s += "\n\n";
                            s += "\nConvolution layer[" + i.ToString(formattingString) + "]\n";
                            for (int j = 0; j < size; j++)
                            {
                                s += "\nFmap [" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].fMaps[j].ToGString();
                                s += "\n";
                                s += "\nFilter[" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].filters[j].ToGString();
                            }
                        }
                    }
                }
            }
            return s;
        }

        public string ToGradientStoreString()
        {
            string s = "";
            int n = cnn.Length;
            for (int i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    s += "\nImage Layer[" + i.ToString(formattingString) + "]\n";
                    s += "\nFmap\n";
                    s += cnn[i].fMaps[0].ToString();

                }
                else
                {
                    if (cnn[i].filters[0].GetType() == typeof(Filters.Connection))
                    {
                        //Connection layer
                        s += "\n\n";
                        s += "\nConnected layer[" + i.ToString(formattingString) + "]\n";
                        for (int l = 0; l < cnn[i].Connections.Count; l++)
                        {
                            s += "\nFmap [" + l.ToString(formattingString) + "]\n";
                            s += cnn[i].fMaps[l].ToGString();
                            s += "\n";
                            s += "\nConnections[" + l.ToString(formattingString) + "]\n";
                            for (int m = 0; m < cnn[i].Connections[0].Count; m++)
                            {
                                s += cnn[i].Connections[l][m].ToGradientStoreString();
                            }
                        }
                    }
                    else
                    {
                        int size = cnn[i].filters.Count;
                        //Convolution or pooling layer
                        if (cnn[i].filters[0].GetType() == typeof(Filters.Maxpool))
                        {
                            s += "\n\n";
                            s += "\nPooling layer[" + i.ToString(formattingString) + "]\n";
                            for (int j = 0; j < size; j++)
                            {
                                s += "\nFmap [" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].fMaps[j].ToGString();
                                s += "\n";
                                s += "\nFilter[" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].filters[j].ToGradientStoreString();
                            }
                        }
                        else
                        {
                            s += "\n\n";
                            s += "\nConvolution layer[" + i.ToString(formattingString) + "]\n";
                            for (int j = 0; j < size; j++)
                            {
                                s += "\nFmap [" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].fMaps[j].ToGString();
                                s += "\n";
                                s += "\nFilter[" + j.ToString(formattingString) + "]\n";
                                s += cnn[i].filters[j].ToGradientStoreString();
                            }
                        }
                    }
                }
            }
            return s;
        }

        public ILayer[] Layer
        {
            get { return cnn;}
        }

        public double Cost
        {
            get { return cost; }
            set { cost = value; }
        }
    }
}