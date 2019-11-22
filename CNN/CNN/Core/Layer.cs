using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NN.Core;

namespace CNN.Core
{
    public class Layer : ILayer
    {
        protected IList<Filter> filter = new List<Filter>();
        protected IList<fMap> fmaps = new List<fMap>();
        protected IList<IList<Filter>> connection = new List<IList<Filter>>();

        public Layer() { }

        public Layer Configure(string config, ILayer input)
        {
            string[] cfg = config.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            string[] tkn = null;

            string cfgstr = "";
            for (int i = 0; i < cfg.Length; i++)
                cfgstr += cfg[i];

            cfg = cfgstr.Split(new char[] { '[', '(', ',', ')', ']' }, StringSplitOptions.RemoveEmptyEntries);

            // syntax: conn(size=7, depth=32, stride=2, fmapsize=144)
            
            string kernel = cfg[0];

            int size = 0, fmapsize = 0, depth = 0, stride = 1, bias = 1; 
            string activation = "relu";

            for (int i = 1; i < cfg.Length; i++)
            {
                tkn = cfg[i].Split(new char[] { '=' }, StringSplitOptions.RemoveEmptyEntries);
                switch (tkn[0])
                {
                    case "depth": // number of filters
                        depth = int.Parse(tkn[1]);
                        break;
                    case "fmapsize": // size of target fmaps
                        fmapsize = int.Parse(tkn[1]);
                        break;
                    case "stride": // stride
                        stride = int.Parse(tkn[1]);
                        break;
                    case "size": // size
                        size = int.Parse(tkn[1]);
                        break;
                    case "bias": // bias
                        bias = int.Parse(tkn[1]);
                        break;
                    case "activation": // activation function
                        activation = tkn[1];
                        break;
                    default:
                        throw new Exception();
                }

            }


            int number_of_fmaps = input.fMaps.Count;
            int index = 0;
            if (cfg[0] == "conn")
            {
                if ((input.filters[0].GetType()==typeof(Filters.Maxpool)) || (input.filters[0].GetType() == typeof(Filters.Convolution)))
                {
                    //if previous layer is pool layer
                    int size_of_fmaps = input.fMaps[0].value.Length;
                    //based on the shape of pooling output squared * number of pooling outputs 
                    int number_of_filters = size_of_fmaps * size_of_fmaps * number_of_fmaps;
                    for (int i = 0; i < depth; i++)
                    {
                        Filter k;
                        fMap w = null;
                        fMap o = null;
                        //output with activation
                        o = new fMap(fmapsize, fmapsize);
                        o.Bias.SetValue(bias, 0);
                        fmaps.Add(o);
                        IList<Filter> fmap_filters = new List<Filter>();
                        //We connect n filters to output o
                        for (int j = 0; j < number_of_filters; j++)
                        {
                            k = Filter.Build(cfg[0], j, size, stride);
                            w = new fMap(fmapsize, fmapsize);
                            k.Source = w;
                            k.Target = o;
                            k.Activation = activation;
                            filter.Add(k);
                            fmap_filters.Add(k);
                        }
                        connection.Add(fmap_filters);
                    }
                    //if (input.filters[0].GetType() == typeof(Filters.Maxpool))
                    //{

                    //}
                }
                else
                {
                    //if the previous layer is a connection layer
                    //the number of filters (weihgts) is equal to the number of nodes in previous layer
                    int number_of_filters =  number_of_fmaps;
                    //for each output node in this layer
                    for (int i = 0; i < depth; i++)
                    {
                        Filter k;
                        fMap o = null;
                        //output with activation
                        o = new fMap(fmapsize, fmapsize);
                        o.Bias.SetValue(bias,0);
                        fmaps.Add(o);
                        IList<Filter> fmap_filters = new List<Filter>();
                        for (int j = 0; j < number_of_filters; j++)
                        {
                            //we connect n filters (W) with n inputs (x) to a single output o
                            k = Filter.Build(cfg[0], j, size, stride);
                            k.Source = input.fMaps[j];
                            k.Target = o;
                            k.Activation = activation;
                            filter.Add(k);
                            fmap_filters.Add(k);
                        }
                      
                        connection.Add(fmap_filters);
                    }
                }

            }
            else
            {
                //Now we are dealing with convolutional and pooling layers only
                    if (number_of_fmaps == 1)
                    {
                    //1 channel image
                    for (int i = 0; i < depth; i++)
                    {
                        Filter k;
                        fMap w = null;
                        w = new fMap(fmapsize, fmapsize);
                        k = Filter.Build(cfg[0], i, size, stride);
                        //Source  feature map is input image
                        k.Source = input.fMaps[0];
                        k.Target = w;
                        fmaps.Add(w);
                        k.bias.SetValue(bias, 0);
                        k.Activation = activation;
                        filter.Add(k);
                        }
                    }

                    if (number_of_fmaps == 3)
                    {
                    //3 channel image (RGB)
                    for (int i = 0; i < depth; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            Filter k;
                            fMap w = null;
                            w = new fMap(fmapsize, fmapsize);
                            k = Filter.Build(cfg[0], j, size, stride);
                            //Source  feature map is input image
                            k.Source = input.fMaps[j];
                            k.Target = w;
                            fmaps.Add(w);
                            k.bias.SetValue(bias, 0);
                            k.Activation = activation;
                            filter.Add(k);
                        }
                    }
                    }

                    else
                    {
                    //source feature map is from previous conv or pool layer
                    for (int i = 0; i < depth; i++)
                    {
                        if (index < number_of_fmaps)
                        {
                            Filter k;
                            fMap w = null;
                            w = new fMap(fmapsize, fmapsize);
                            k = Filter.Build(cfg[0], i, size, stride);
                            k.Source = input.fMaps[index];
                            k.Target = w;
                            k.bias.SetValue(bias, 0);
                            k.Activation = activation;
                            filter.Add(k);
                            fmaps.Add(w);
                            index++;
                        }
                    }
                    }
            }

            return this;
        }
        /// <summary>
        /// returns list of feature maps
        /// </summary>
        public IList<fMap> fMaps
        {
            get { return fmaps; }
        }

        public IList<Filter> filters
        {
            get { return filter; }
        }

        public IList<IList<Filter>> Connections
        {
            get { return connection; }
        }

    }
}