using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    public class Image : ILayer
    {
        protected IList<fMap> fmaps = new List<fMap>();
        protected IList<Filter> filter = new List<Filter>();
        Filter k;

        public Image() { }

        public Image(int rows, int cols) 
        {
            Configure(rows, cols);
        }

        public fMap Blue
        {
            get { return fmaps[2]; }
        }

        public Image Configure(int rows, int cols)
        {
            for (int i = 0; i < 1; i++)
                fmaps.Add(new fMap(rows, cols));

            k = Filter.Build("image", 1, rows, cols,1,1);
            filter.Add(k);
            return this;
        }

        /// <summary>
        /// returns list of feature maps
        /// </summary>
        public IList<fMap> fMaps
        {
            get { return fmaps; }
        }

        public fMap Green
        {
            get { return fmaps[1]; }
        }

        public fMap Red
        {
            get { return fmaps[0]; }
        }

        public IList<Filter> filters
        {
            get { return filter; }
        }

        IList<IList<Filter>> ILayer.Connections => null;

    }
}