# 4--6 Week Forecast-Guided Redistribution Pilot

## 1. Decision to test

Test whether forecast-guided redistribution improves bike availability
during predictable demand peaks compared with the operator's current
redistribution policy.

**Do not frame this as proof that redistribution is already the root
cause.** The historical UCI dataset contains citywide completed rentals,
not station stock-outs, failed rental attempts, station flows or trip
durations. Those operational data must be collected during the pilot.

## 2. Pilot hypothesis

-   **Primary hypothesis:** Forecast-guided redistribution reduces
    empty-station minutes during pre-defined peak windows compared with
    business-as-usual operations.
-   **Secondary hypotheses:** It increases completed rentals and reduces
    the imbalance between stations with persistent surpluses and
    stations with persistent deficits.
-   **Guardrail hypothesis:** The improvement is achieved without an
    unacceptable increase in driver hours, kilometres, operating cost,
    unsafe loading or customer complaints.

## 3. Duration and phases

  ------------------------------------------------------------------------
  Phase                                     Duration Activities
  --------------------- ---------------------------- ---------------------
  Instrumentation and                         Week 1 Validate station
  baseline                                           inventory snapshots,
                                                     failed-rental
                                                     logging, bike health
                                                     and driver logs.
                                                     Record
                                                     business-as-usual
                                                     operations in all
                                                     pilot zones.

  Controlled pilot      Weeks 2--4 (4-week total) or Run forecast-guided
                           Weeks 2--6 (6-week total) redistribution in
                                                     treatment clusters;
                                                     matched control
                                                     clusters continue the
                                                     current policy.

  Close-out              Final 2--3 days, within the Reconcile data,
                                    final pilot week document deviations,
                                                     estimate impact and
                                                     operating cost.
  ------------------------------------------------------------------------

If instrumentation cannot be validated in Week 1, delay randomisation
rather than start with unreliable outcome data. Confirm before launch
that the selected duration and number of clusters provide enough
observations to detect an operationally meaningful effect. If the
available number of independent clusters is too small, treat the pilot
as exploratory rather than presenting it as a definitive causal test.

## 4. Experimental design

### Preferred design: matched station-cluster randomisation

1.  Group nearby stations into operational clusters to reduce
    bike-transfer contamination across treatment and control.
2.  Match clusters using baseline demand, empty-station minutes, station
    count, geography and operating hours.
3.  Randomly assign one cluster in each pair to:
    -   **Treatment:** forecast-guided redistribution.
    -   **Control:** existing redistribution schedule and rules.
4.  Keep assignment fixed for the pilot. Log every cross-cluster
    transfer; avoid transfers between arms where operationally possible.
5.  Compare changes from baseline between treatment and control
    (difference-in-differences), with uncertainty intervals clustered at
    the randomisation unit.

If there are too few independent clusters for a credible matched design,
consider a pre-registered switchback by cluster-week, randomising
treatment weeks and allowing a defined washout/clearance period. Do not
switch ad hoc in response to observed results. The switchback design
should account for carryover effects from bikes moved in a previous
treatment period.

## 5. Treatment policy

Use the hourly demand profile to choose *when* to consider positioning
bikes, then use live station data to decide *where*.

-   Working days: test a morning positioning round completed before
    approximately 07:00 and an afternoon round completed before
    approximately 16:30. These are starting hypotheses based on citywide
    historical peaks, not an optimised schedule.
-   Non-working days: use a separate profile; test midday-oriented
    positioning if station-level forecasts support it.
-   Each dispatch should use forecast demand by station, current
    rentable stock, expected returns, dock capacity, bike health, driver
    capacity and travel time.
-   Do not move bikes based only on citywide hourly totals.
-   Record the forecast version and operational inputs used for each
    dispatch so the decision can be audited later.

## 6. Required data collection

  -----------------------------------------------------------------------
  Data                    Minimum fields          Frequency
  ----------------------- ----------------------- -----------------------
  Station inventory       station_id, timestamp,  Every 5--15 minutes
                          rentable_bikes,         
                          empty_docks,            
                          out_of_service_bikes    

  Trips                   trip_id, start_station, Each trip
                          end_station,            
                          start_time, end_time    

  Failed/abandoned        station_id, timestamp,  Event-level
  rentals                 failure reason          

  Forecasts               station_id,             Every forecast run
                          forecast_created_at,    
                          target_time, predicted  
                          rentals, uncertainty    
                          interval                

  Rebalancing             vehicle/driver, origin, Every move
                          destination, bikes      
                          moved,                  
                          departure/arrival time, 
                          distance, cost          

  Bike health             bike_id or              At status change
                          station-level           
                          unavailable count,      
                          reason, repair time     

  Context                 weather, holidays,      Hourly/event-level
                          events, service         
                          disruptions             
  -----------------------------------------------------------------------

Use consistent station IDs and timestamps. Preserve the forecast as it
was available at decision time; do not overwrite old forecasts with
later revisions. Define data access, retention and any necessary
protection for trip- and driver-level records before collection.

## 7. Success metrics

### Primary outcome

**Empty-station minutes per station-hour during defined peak windows.**

Define an empty station in advance (for example, zero rentable bikes at
an inventory snapshot). Use a consistent rule to estimate the duration
between snapshots---for example, carry the observed state forward only
until the next valid snapshot, and flag gaps longer than the pre-defined
tolerance as missing rather than assuming the station remained empty.
Calculate the metric using the same rule in treatment and control.

### Secondary outcomes

-   Failed rental attempts per 1,000 attempted rentals.
-   Completed rentals per station-hour.
-   Percentage of station-hours with stock within the pre-defined target
    band.
-   Idle-bike hours at stations above the surplus threshold.
-   Number and duration of station stock-out events.
-   Complaints related to bike unavailability.

### Operational cost and guardrails

-   Driver hours and kilometres per 100 bikes moved.
-   Cost per additional successful rental.
-   Rebalancing moves and bikes moved per shift.
-   Safety incidents, damaged bikes and missed service-level
    commitments.
-   Control-zone service quality (watch for contamination or resource
    diversion).

Report absolute and relative changes, sample sizes, confidence intervals
and operational cost---not p-values alone.

## 8. Analysis plan

Pre-register the primary outcome, peak windows, treatment assignment,
exclusion rules, missing-data handling and analysis before reviewing
pilot outcomes.

Estimate:

`Treatment effect = (Treatment_post − Treatment_baseline) − (Control_post − Control_baseline)`

Use station-clustered uncertainty estimates because observations within
a station/cluster are not independent. Adjust for day-of-week and
calendar effects if pre-specified. Report sensitivity analyses for
missing inventory snapshots and unusual disruptions.

Do not claim success if the primary metric improves only because
treatment stations received more driver hours or control stations were
deprived of resources; report those costs and any resource transfers.
Interpret the result cautiously if the pilot has few independent
clusters or wide uncertainty intervals.

## 9. Decision rules

Agree thresholds with the operations team before launch. Suggested
decision logic:

-   **Scale cautiously:** The primary metric improves by a pre-agreed
    operationally meaningful amount, uncertainty is consistent with
    benefit, and cost/guardrails remain acceptable.
-   **Refine and retest:** The direction is positive but uncertain, or
    benefits are concentrated in certain station types or time windows.
-   **Do not scale:** There is no meaningful improvement, operating cost
    rises beyond the agreed limit, or safety/service guardrails worsen.
-   **Consider targeted fleet additions only after:** Reliable stock-out
    evidence persists after rebalancing and repairs, and unmet demand is
    measured at the affected stations.

## 10. What the current dataset can and cannot do

**Can support:** Citywide hourly timing patterns,
working-day/non-working-day differences, historical completed-rental
percentiles and scenario forecasts.

**Cannot support:** Station donor/receiver identification, actual
trip-duration estimates, stock-out minutes, failed-rental counts, route
optimisation or causal claims about redistribution.

The pilot is designed to collect the missing operational evidence needed
to make those decisions.
