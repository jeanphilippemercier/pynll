from loguru import logger
import json as json
from uquake.core.event import ResourceIdentifier
from obspy.core import UTCDateTime


def attribdict_to_dict(attrib_dict):
    """
    convert AttribDict to dict
    :param attrib_dict: an attribute dictionary
    :type attrib_dict: obspy.core.event.AttribDict
    :return:
    """
    out_dict = {}
    for key in attrib_dict.keys():
        if hasattr(attrib_dict[key], 'keys'):
            dict_items = attribdict_to_dict(attrib_dict[key])
            out_dict[key] = dict_items
        elif type(attrib_dict[key]) is ResourceIdentifier:
            out_dict[key] = str(attrib_dict[key])
        elif type(attrib_dict[key]) is UTCDateTime:
            out_dict[key] = attrib_dict[key].isoformat()
        else:
            out_dict[key] = attrib_dict[key]

    return out_dict


def attribdict_to_json(attrib_dict):
    return json.dumps(attribdict_to_dict(attrib_dict))

def quantity_error_to_dict(quantity_error, output_json=True):
    quantity_error_dict = {}
    for key in quantity_error.__dict__.keys():
        quantity_error_dict[key] = quantity_error[key]

    if output_json:
        return json.load(quantity_error_dict)

    else:
        return quantity_error_dict


# def pick_to_dict(picks, output_json=True):
#     pick_list = []
#     for pick in picks:
#         out_pick = pick.__dict__.copy()
#         for key in pick.__dict__.keys()
#             if type(pick[key]) is AttribDict
#         if output_json:
#             pick_list.append(out_pick)
#         else:
#             pick_list.append(json.dumps(out_pick))
#
#     return pick_list




def event_to_arrivals_dict(event, preferred_origin=True, json=True):

    arrivals = []
    origin = event.origins[-1]

    if preferred_origin:
        origin = event.preferred_origin()
        if origin is None:
            logger.warning('preferred origin is not set, origin[-1] will'
                           'be used instead')
    for arrival in origin.arrivals:

        dict_out = {'arrival_id': arrival.resource_id,
                    'pick_id': arrival.pick_id,
                    'phase': arrival.phase,
                    'time_correction': arrival.time_correction,
                    'azimuth': arrival.azimuth,
                    'distance': arrival.distance,
                    'takeoff_angle': arrival.takeoff_angle,
                    'takeoff_angle_error': {
                        'uncertainty': arrival.takeoff_angle_error.uncertainty,
                        'lower_uncertainty':
                            arrival.takeoff_angle_error.lower_uncertaity,
                        'upper_uncertainty':
                            arrival.takeoff_angle_error.upper_uncertainty
                    }}


        pick = arrival.get_pick()

        if pick.evaluation_status == 'rejected':
            continue

        sensor = pick.sensor
        instrument_identification = pick.waveform_id.channel_code[0:2]
        component = pick.waveform_id.channel_code[-1]
        phase_onset = 'e' if pick.onset in ['emergent', 'questionable'] \
            else 'i'
        phase_descriptor = arrival.phase.upper()
        if pick.polarity is None:
            first_motion = '?'
        else:
            first_motion = 'U' if pick.polarity.lower() == 'positive' \
                else 'D'
        datetime_str = pick.time.strftime('%Y%m%d %H%M %S.%f')

        error_type = 'GAU'
        if arrival.phase.upper() == 'P':
            pick_error = f'{self.p_pick_error:0.2e}'
        else:
            pick_error = f'{self.s_pick_error:0.2e}'

        # not implemented
        coda_duration = -1
        amplitude = -1
        period = -1
        phase_weight = 1