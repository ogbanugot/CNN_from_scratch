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

        IList<Filter> ILayer.filters => throw new NotImplementedException();
        IList<IList<Filter>> ILayer.Connections => throw new NotImplementedException();

    }
}