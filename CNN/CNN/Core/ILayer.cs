using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    public interface ILayer
    {
        IList<fMap> fMaps { get; }
        IList<Filter> filters { get; }
        IList<IList<Filter>> Connections { get; }
    }
}
