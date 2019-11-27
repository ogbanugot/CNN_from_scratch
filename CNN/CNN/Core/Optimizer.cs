using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    class Optimizer
    {
        public Model AdamGD(List<double[][]> batch_image, List<double[][]> batch_label, Model cnn, double learning_rate, double beta1, double beta2, int batch_size)
        {   //Adam gradient decent
            //returns optimized parameters and cost
            //Initialize gradient_storage to zero
            int n = cnn.Layer.Length;
            for (int i = n - 1; i >= 1; i--)
            {
                int size = cnn.Layer[i].filters.Count;
                //int no = 0;
                //output layer
                if (i == n - 1 && cnn.Layer[i].filters[0].GetType() == typeof(Filters.Connection))
                {
                    //for each fmap reset gradient storage in bias
                    for (int j = 0; j < cnn.Layer[i].fMaps.Count; j++)
                    {
                        cnn.Layer[i].fMaps[j].Bias[2] = 0;

                    }
                    //reset storage derivative of weights
                    //for all connections
                    for (int h = 0; h < cnn.Layer[i].Connections.Count; h++)
                    {
                        //for each weight in connection
                        for (int m = 0; m < cnn.Layer[i].Connections[0].Count; m++)
                        {
                            cnn.Layer[i].Connections[h][m].GradientStorageReset = 0;
                        }
                    }

                }

                if ((i != n - 1) && (cnn.Layer[i].filters[0].GetType() == typeof(Filters.Connection)))
                {
                    //Hidden layer
                    //for each fmap in hidden layer
                    for (int j = 0; j < cnn.Layer[i].fMaps.Count; j++)
                    {
                        cnn.Layer[i].fMaps[j].Bias[2] = 0;

                    }
                    //reset storage derivative of weights
                    //for all connections
                    for (int h = 0; h < cnn.Layer[i].Connections.Count; h++)
                    {
                        //for each weight in connection
                        for (int m = 0; m < cnn.Layer[i].Connections[0].Count; m++)
                        {
                            cnn.Layer[i].Connections[h][m].GradientStorageReset = 0;
                        }
                    }
                }

                if ((i != n - 1) && (cnn.Layer[i].filters[0].GetType() == typeof(Filters.Convolution)))
                {
                    //Reset gradient for filters on convolution layers
                    for (int j = 0; j < cnn.Layer[i].fMaps.Count; j++)
                    {
                        cnn.Layer[i].filters[j].GradientStorageReset = 0;
                        cnn.Layer[i].filters[j].bias[2] = 0;

                    }
                }

            }

            //Now we loop over the images in a batch
            //We shuffle the batch
            Linalg linalg = new Linalg();
            List<double[][]> images = new List<double[][]>();
            List<double[][]> labels = new List<double[][]>();
            List<List<double[][]>> result = new List<List<double[][]>>();
            result = linalg.Shuffle(batch_image, batch_label);
            Loss loss = new Loss();
            double ls = 0; //loss
            images = result[0];
            labels = result[1];
            double[][] probs = new double[1][];
            for (int l = 0; l < probs.Length; l++)
                probs[l] = new double[cnn.Layer[cnn.Layer.Length - 1].fMaps.Count];
            //for each image in the batch

            //string s1 = cnn.ToString();
            ////string g1 = cnn.ToGString();
            //System.IO.File.WriteAllText("/home/ugot/mnist/string", s1);
            ////System.IO.File.WriteAllText("/home/ugot/mnist/gstring", g1);
            //string g5 = cnn.ToGradientStoreString();
            //System.IO.File.WriteAllText("/home/ugot/mnist/grdstring", g5);

            for (int i=0; i<images.Count; i++)
            {

                //Console.WriteLine("Before forward Propagation");
                //string s1 = cnn.ToString();
                ////string g1 = cnn.ToGString();
                //System.IO.File.WriteAllText("/home/ugot/mnist/string", s1);
                ////System.IO.File.WriteAllText("/home/ugot/mnist/gstring", g1);
                //string g5 = cnn.ToGradientStoreString();
                //System.IO.File.WriteAllText("/home/ugot/mnist/grdstring", g5);

                cnn.Forward(images[i]);
                //Calculate loss
                for (int m=0; m< cnn.Layer[cnn.Layer.Length - 1].fMaps.Count; m++)
                {
                    probs[0][m] = cnn.Layer[cnn.Layer.Length - 1].fMaps[m].value[0][0].Value;
                }
                ls += loss.CategoricalCrossEntropy(probs, labels[i]);

                //Console.WriteLine("After forward Propagation");
                //string s2 = cnn.ToString();
                //System.IO.File.WriteAllText("/home/ugot/mnist/string", s2);

                cnn.Backward(labels[i]);

                //Console.WriteLine("After backward propagation");
                //string g3 = cnn.ToGString();
                //System.IO.File.WriteAllText("/home/ugot/mnist/gstring", g3);
                //string g5 = cnn.ToGradientStoreString();
                //System.IO.File.WriteAllText("/home/ugot/mnist/grdstring", g5);

            }
            //Adam weight update
            //we have to go over the model layers and calculate momentum and RMS prop (ADAM) parameters, adjust parameters.
            /* 
            v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
            s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
            f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam

            bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
            bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
            b1 -= lr * bv1/np.sqrt(bs1+1e-7)
            */

            for (int i = n - 1; i >= 1; i--)
            {
                int size = cnn.Layer[i].filters.Count;
                //output layer
                if (i == n - 1 && cnn.Layer[i].filters[0].GetType() == typeof(Filters.Connection))
                {
                    for (int j = 0; j < cnn.Layer[i].fMaps.Count; j++)
                    {
                        double bv = 0;
                        double bs = 0;
                        bv = (beta1 * bv) + (1-beta1)*(cnn.Layer[i].fMaps[j].Bias[2]/batch_size);
                        bs = (beta1 * bs) + (1 - beta2)*Math.Pow(cnn.Layer[i].fMaps[j].Bias[2] / batch_size, 2);
                        cnn.Layer[i].fMaps[j].Bias[0] -= learning_rate * (bv / Math.Sqrt(bs + 1e-7));


                    }
                    //reset storage derivative of weights
                    //for all connections
                    for (int h = 0; h < cnn.Layer[i].Connections.Count; h++)
                    {
                        //for each weight in connection
                        for (int m = 0; m < cnn.Layer[i].Connections[0].Count; m++)
                        {
                            double wv = 0;
                            double ws = 0;
                            wv = (beta1 * wv) + (1 - beta1) * (cnn.Layer[i].Connections[h][m].GradientStorage[0][0] / batch_size);
                            ws = (beta1 * ws) + (1 - beta2) * Math.Pow(cnn.Layer[i].Connections[h][m].GradientStorage[0][0] / batch_size, 2);
                            double gradient = learning_rate * (wv / Math.Sqrt(ws + 1e-7));
                            double diff = cnn.Layer[i].Connections[h][m].value[0][0].Value - gradient;
                            cnn.Layer[i].Connections[h][m].value[0][0].Value = diff;
                        }
                    }

                }

                if ((i != n - 1) && (cnn.Layer[i].filters[0].GetType() == typeof(Filters.Connection)))
                {
                    //Hidden layer
                    //for each fmap in hidden layer
                    for (int j = 0; j < cnn.Layer[i].fMaps.Count; j++)
                    {
                        double bv = 0;
                        double bs = 0;
                         bv = (beta1 * bv) + (1 - beta1) * (cnn.Layer[i].fMaps[j].Bias[2] / batch_size);
                         bs = (beta1 * bs) + (1 - beta2) * Math.Pow(cnn.Layer[i].fMaps[j].Bias[2] / batch_size, 2);
                        cnn.Layer[i].fMaps[j].Bias[0] -= learning_rate * (bv / Math.Sqrt(bs + 1e-7));

                    }
                    //reset storage derivative of weights
                    //for all connections
                    for (int h = 0; h < cnn.Layer[i].Connections.Count; h++)
                    {
                        //for each weight in connection
                        for (int m = 0; m < cnn.Layer[i].Connections[0].Count; m++)
                        {
                            double wv = 0;
                            double ws = 0;
                             wv = (beta1 * wv) + (1 - beta1) * (cnn.Layer[i].Connections[h][m].GradientStorage[0][0] / batch_size);
                             ws = (beta1 * ws) + (1 - beta2) * Math.Pow(cnn.Layer[i].Connections[h][m].GradientStorage[0][0] / batch_size, 2);
                            double gradient = learning_rate * (wv / Math.Sqrt(ws + 1e-7));
                            double diff = cnn.Layer[i].Connections[h][m].value[0][0].Value - gradient;
                            cnn.Layer[i].Connections[h][m].value[0][0].Value = diff;
                        }
                    }
                }

                if ((i != n - 1) && (cnn.Layer[i].filters[0].GetType() == typeof(Filters.Convolution)))
                {
                    for (int j = 0; j < cnn.Layer[i].fMaps.Count; j++)
                    {
                        for (int l = 0; l < cnn.Layer[i].filters[j].Size; l++)
                        {
                            for (int m = 0; m < cnn.Layer[i].filters[j].Size; m++)
                            {
                                double wv = 0;
                                double ws = 0 ;
                                wv = (beta1 * wv) + (1 - beta1) * (cnn.Layer[i].filters[j].GradientStorage[l][m] / batch_size);
                                ws = (beta1 * ws) + (1 - beta2) * Math.Pow(cnn.Layer[i].filters[j].GradientStorage[l][m] / batch_size, 2);
                                double gradient = learning_rate * (wv / Math.Sqrt(ws + 1e-7));
                                double diff = cnn.Layer[i].filters[j].value[l][m].Value - gradient;
                                cnn.Layer[i].filters[j].value[l][m].Value = diff;
                            }

                        }
                        double bv = 0;
                        double bs = 0;
                         bv = (beta1*bv) + (1 - beta1) * (cnn.Layer[i].filters[j].bias[2] / batch_size);
                         bs = (beta1 * bs)+ (1 - beta2) * Math.Pow(cnn.Layer[i].filters[j].bias[2] / batch_size, 2);
                        cnn.Layer[i].filters[j].bias[0] -= learning_rate * (bv / Math.Sqrt(bs + 1e-7));
                    }
                }

            }

            //Console.WriteLine("After Weight Adjustment");
            //string s4 = cnn.ToString();
            //string g4 = cnn.ToGString();
            //string g6 = cnn.ToGradientStoreString();
            //System.IO.File.WriteAllText("/home/ugot/mnist/grdstring2", g6);
            //System.IO.File.WriteAllText("/home/ugot/mnist/string2", s4);
            //System.IO.File.WriteAllText("/home/ugot/mnist/gstring2", g4);

            //set cost for batch
            cnn.Cost = ls / batch_size;
            return cnn;
        }

    }
}
