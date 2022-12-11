
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables

def smart_cond(pred, true_fn=None, false_fn=None, name=None):
    """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

    If `pred` is a bool or has a constant value, we return either `true_fn()`
    or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

    Arguments:
        pred: A scalar determining whether to return the result of `true_fn` or
            `false_fn`.
        true_fn: The callable to be performed if pred is true.
        false_fn: The callable to be performed if pred is false.
        name: Optional name prefix when using `tf.cond`.

    Returns:
        Tensors returned by the call to either `true_fn` or `false_fn`.

    Raises:
        TypeError: If `true_fn` or `false_fn` is not callable.
    """
    if not callable(true_fn):
        raise TypeError("`true_fn` must be callable.")
    if not callable(false_fn):
        raise TypeError("`false_fn` must be callable.")

    pred_value = smart_constant_value(pred)
    if pred_value is not None:
        if pred_value:
            return true_fn()
        else:
            return false_fn()
    else:
        return control_flow_ops.cond(pred, true_fn=true_fn, false_fn=false_fn,
                                                                 name=name)


def smart_constant_value(pred):
    """Return the bool value for `pred`, or None if `pred` had a dynamic value.

    Arguments:
        pred: A scalar, either a Python bool or tensor.

    Returns:
        True or False if `pred` has a constant boolean value, None otherwise.

    Raises:
        TypeError: If `pred` is not a Tensor or bool.
    """
    if isinstance(pred, ops.Tensor):
        pred_value = tensor_util.constant_value(pred)
        # TODO(skyewm): consider folding this into tensor_util.constant_value.
        # pylint: disable=protected-access
        if pred_value is None:
            pred_value = c_api.TF_TryEvaluateConstant_wrapper(pred.graph._c_graph,
                                                                                                                pred._as_tf_output())
        # pylint: enable=protected-access
    elif pred in (0, 1):  # Accept 1/0 as valid boolean values
        pred_value = bool(pred)
    elif isinstance(pred, bool):
        pred_value = pred
    else:
        raise TypeError("`pred` must be a Tensor, or a Python bool, or 1 or 0. "
                                        "Found instead: %s" % type(pred))

    return pred_value

def constant_value(pred):
    """Return the bool value for `pred`, or None if `pred` had a dynamic value.

    Arguments:
        pred: A scalar, either a Python bool or a TensorFlow boolean variable
            or tensor, or the Python integer 1 or 0.

    Returns:
        True or False if `pred` has a constant boolean value, None otherwise.

    Raises:
        TypeError: If `pred` is not a Variable, Tensor or bool, or Python
            integer 1 or 0.
    """
    # Allow integer booleans.
    if isinstance(pred, int):
        if pred == 1:
            pred = True
        elif pred == 0:
            pred = False

    if isinstance(pred, variables.Variable):
        return None
    return smart_constant_value(pred)